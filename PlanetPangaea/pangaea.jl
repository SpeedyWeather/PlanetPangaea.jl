using CairoMakie
using Colors
using Images
using ImageFiltering
using NCDatasets
using NearestNeighbors
using Statistics

const lat_size, lon_size = 192, 384
const image_path = joinpath(@__DIR__, "../data/input_images/olenekian_pangaea.png")

# Try loading the image, fallback to zeros
img = try
    RGB.(load(image_path))
catch e
    println("Error loading image. Using fallback zeros.")
    zeros(RGB{Float32}, lat_size, lon_size)
end

# Resize using Bilinear/Lanczos approximation
img_resized = imresize(img, (lat_size, lon_size))

# Define colors (RGB values mapped 0.0 to 1.0)
colors_dict = Dict(
    "ocean_deep" => RGB(25 / 255, 60 / 255, 100 / 255),
    "ocean_shelf" => RGB(100 / 255, 180 / 255, 220 / 255),
    "bg_grey" => RGB(200 / 255, 200 / 255, 200 / 255),
    "desert_mtn" => RGB(180 / 255, 140 / 255, 90 / 255),
    "low_veg" => RGB(120 / 255, 140 / 255, 80 / 255),
    "high_veg" => RGB(40 / 255, 80 / 255, 40 / 255),
    "ice_clouds" => RGB(230 / 255, 240 / 255, 250 / 255)
)
color_keys = collect(keys(colors_dict))
color_values = collect(values(colors_dict))

# Classify grid based on minimum color distance
classified_grid = zeros(Int, lat_size, lon_size)
for j in 1:lon_size
    for i in 1:lat_size
        c = img_resized[i, j]
        dists = [colordiff(c, ref_c) for ref_c in color_values]
        classified_grid[i, j] = argmin(dists)
    end
end

base_veg_map = zeros(Int, lat_size, lon_size)
base_orog = zeros(Float64, lat_size, lon_size)

for j in 1:lon_size
    for i in 1:lat_size
        c_name = color_keys[classified_grid[i, j]]
        if c_name in ["ocean_deep", "ocean_shelf", "bg_grey"]
            base_veg_map[i, j] = 0
            base_orog[i, j] = 0.0
        elseif c_name in ["desert_mtn", "ice_clouds"]
            base_veg_map[i, j] = 1
            base_orog[i, j] = 0.8
        elseif c_name == "low_veg"
            base_veg_map[i, j] = 2
            base_orog[i, j] = 0.4
        elseif c_name == "high_veg"
            base_veg_map[i, j] = 3
            base_orog[i, j] = 0.2
        end
    end
end

# Clean Edges and Despeckle
y_norm = range(1, -1, length = lat_size)
x_norm = range(-1, 1, length = lon_size)
yv = [y for y in y_norm, x in x_norm]
xv = [x for y in y_norm, x in x_norm]

mollweide_mask = (xv .^ 2 .+ yv .^ 2) .<= 0.98
base_veg_map[.!mollweide_mask] .= 0
base_orog[.!mollweide_mask] .= 0.0

# Median filter of size 3x3
base_veg_map = mapwindow(median, base_veg_map, (3, 3))

# Core Land-Sea Mask
lsm_map = Int.(base_veg_map .> 0)

function stabilize_field(data::Matrix{Float64}, mask::BitMatrix, sigma::Float64)
    filled_data = copy(data)
    invalid = .!mask

    if any(invalid) && any(mask)
        valid_coords = findall(mask)
        invalid_coords = findall(invalid)

        # Build KDTree for fast nearest-neighbor lookup
        tree_points = hcat([[c.I[1], c.I[2]] for c in valid_coords]...)
        query_points = hcat([[c.I[1], c.I[2]] for c in invalid_coords]...)

        tree = KDTree(Float64.(tree_points))
        idxs, _ = knn(tree, Float64.(query_points), 1)

        for (i, query_idx) in enumerate(invalid_coords)
            nearest_valid_idx = valid_coords[idxs[i][1]]
            filled_data[query_idx] = data[nearest_valid_idx]
        end
    end

    # Apply Gaussian smoothing with reflection boundary conditions
    return imfilter(filled_data, Kernel.gaussian(sigma), "reflect")
end

orography_map = imfilter(base_orog, Kernel.gaussian(3.0), "reflect") .* 3500.0
orography_map[lsm_map .== 0] .= 0.0

raw_vegh = Float64.(base_veg_map .== 3)
raw_vegl = Float64.(base_veg_map .== 2)

vegh_map = stabilize_field(raw_vegh, lsm_map .== 1, 1.5)
vegl_map = stabilize_field(raw_vegl, lsm_map .== 1, 1.5)

total_veg = vegh_map .+ vegl_map
overflow = total_veg .> 1.0
vegh_map[overflow] .= vegh_map[overflow] ./ total_veg[overflow]
vegl_map[overflow] .= vegl_map[overflow] ./ total_veg[overflow]

raw_swl1 = zeros(Float64, lat_size, lon_size)
raw_swl1[base_veg_map .== 1] .= 0.05
raw_swl1[base_veg_map .== 2] .= 0.2
raw_swl1[base_veg_map .== 3] .= 0.6
raw_swl1[base_veg_map .== 0] .= mean(raw_swl1[base_veg_map .> 0])

swl1_map = stabilize_field(raw_swl1, lsm_map .== 1, 2.0)
swl2_map = clamp.(swl1_map .* 1.2, 0.0, 1.0)

raw_albedo = zeros(Float64, lat_size, lon_size)
raw_albedo[base_veg_map .== 1] .= 0.35
raw_albedo[base_veg_map .== 2] .= 0.2
raw_albedo[base_veg_map .== 3] .= 0.12
albedo_map = stabilize_field(raw_albedo, lsm_map .== 1, 1.5)

# Ocean Albedo formula
lat_rad = range(pi / 2, -pi / 2, length = lat_size)
lat_grid_rad = [lat for lat in lat_rad, lon in range(-180, 180, length = lon_size)]

ocean_albedo = 0.06 ./ (cos.(lat_grid_rad) .+ 0.15)
albedo_map[lsm_map .== 0] .= ocean_albedo[lsm_map .== 0]

sst_map = 10.0 .+ 25.0 .* cos.(lat_grid_rad)
sst_map = stabilize_field(sst_map, lsm_map .== 0, 2.0)

# Re-masking
vegh_map[lsm_map .== 0] .= 0.0
vegl_map[lsm_map .== 0] .= 0.0
swl1_map[lsm_map .== 0] .= NaN
swl2_map[lsm_map .== 0] .= NaN

# --- 3D Seasonal Expansions ---
const time_steps = 12
t_array = reshape(0:11, 1, 1, time_steps)

phase_map = ifelse.(lat_grid_rad .> 0, -pi / 2, pi / 2)
amplitude = 0.15
seasonal_multiplier = 1.0 .+ amplitude .* sin.((2 * pi .* t_array ./ time_steps) .+ phase_map)

swl1_3d = clamp.(swl1_map .* seasonal_multiplier, 0.0, 1.0)
swl2_3d = clamp.(swl2_map .* seasonal_multiplier, 0.0, 1.0)

ocean_mask_3d = repeat(lsm_map .== 0, 1, 1, time_steps)
swl1_3d[ocean_mask_3d] .= NaN
swl2_3d[ocean_mask_3d] .= NaN

sst_map[lsm_map .== 1] .= NaN
max_sst_amplitude = 4.0
sst_amplitude_map = max_sst_amplitude .* abs.(sin.(lat_grid_rad))

sst_3d = sst_map .+ sst_amplitude_map .* sin.((2 * pi .* t_array ./ time_steps) .+ phase_map)
sst_3d = max.(sst_3d, 0.0)

land_mask_3d = repeat(lsm_map .== 1, 1, 1, time_steps)
sst_3d[land_mask_3d] .= NaN

function export_dataset_to_netcdf(variables_dict, description, filename)
    lat_deg = collect(range(90, -90, length = lat_size))
    lon_deg = collect(range(-180, 180, length = lon_size))
    time_coords = collect(0:11)

    NCDataset(filename, "c") do ds
        ds.attrib["description"] = description

        defDim(ds, "lon", lon_size)
        defDim(ds, "lat", lat_size)

        v_lon = defVar(ds, "lon", Float64, ("lon",))
        v_lon[:] = lon_deg
        v_lon.attrib["units"] = "degrees_east"
        v_lon.attrib["standard_name"] = "longitude"

        v_lat = defVar(ds, "lat", Float64, ("lat",))
        v_lat[:] = lat_deg
        v_lat.attrib["units"] = "degrees_north"
        v_lat.attrib["standard_name"] = "latitude"

        is_3d = any(ndims(info["data"]) == 3 for (_, info) in variables_dict)
        if is_3d
            defDim(ds, "time", length(time_coords))
            v_time = defVar(ds, "time", Float64, ("time",))
            v_time[:] = time_coords
            v_time.attrib["units"] = "months"
        end

        for (var_name, info) in variables_dict
            data = info["data"]
            dims = ndims(data) == 3 ? ("lon", "lat", "time") : ("lon", "lat")

            data_permuted = ndims(data) == 3 ? permutedims(data, (2, 1, 3)) : permutedims(data, (2, 1))

            v = defVar(
                ds,
                var_name,
                eltype(data_permuted),
                dims;
                fillvalue = NaN,
                deflatelevel = 5,
                shuffle = true
            )
            v[:] = data_permuted
            v.attrib["standard_name"] = info["std_name"]
            v.attrib["units"] = info["units"]
        end
    end
    return println("Saved: $filename")
end


outpath = joinpath(@__DIR__, "../data/boundary_conditions/")

export_dataset_to_netcdf(
    Dict("lsm" => Dict("data" => Float64.(lsm_map), "std_name" => "land_binary_mask", "units" => "1")),
    "Land-Sea Mask", joinpath(outpath, "lsm.nc")
)
export_dataset_to_netcdf(
    Dict("orog" => Dict("data" => orography_map, "std_name" => "surface_altitude", "units" => "m")),
    "Surface Orography Approximation", joinpath(outpath, "orography.nc")
)
export_dataset_to_netcdf(
    Dict(
        "vegh" => Dict("data" => vegh_map, "std_name" => "area_fraction_of_high_vegetation", "units" => "1"),
        "vegl" => Dict("data" => vegl_map, "std_name" => "area_fraction_of_low_vegetation", "units" => "1")
    ),
    "Vegetation Fractions", joinpath(outpath, "vegetation.nc")
)
export_dataset_to_netcdf(
    Dict(
        "swl1" => Dict("data" => swl1_3d, "std_name" => "volume_fraction_of_water_in_soil_layer_1", "units" => "m3 m-3"),
        "swl2" => Dict("data" => swl2_3d, "std_name" => "volume_fraction_of_water_in_soil_layer_2", "units" => "m3 m-3")
    ),
    "Volumetric Soil Moisture in Layers (12-Month Climatology)", joinpath(outpath, "soil_moisture.nc")
)
export_dataset_to_netcdf(
    Dict("sst" => Dict("data" => sst_3d .+ 273, "std_name" => "sea_surface_temperature", "units" => "degC")),
    "Synthetic Sea Surface Temperature (12-Month Climatology)", joinpath(outpath, "sst.nc")
)
export_dataset_to_netcdf(
    Dict("alb" => Dict("data" => albedo_map, "std_name" => "surface_albedo", "units" => "1")),
    "Surface Albedo", joinpath(outpath, "albedo.nc")
)

# Initialize the figure canvas (scaled slightly down from plt.subplots(figsize=(24,10)) for screen viewing)
fig = Figure(size = (2000, 900))

# Define a helper function to replicate the tight grid and colorbar placement
function plot_panel!(position, data, title_str, cmap, crange; is_categorical = false)
    # Use DataAspect to keep the map ratio square/consistent, mimicking imshow
    ax = CairoMakie.Axis(position[1, 1], title = title_str, aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)

    # rotr90 forces the Julia matrix into the top-down visual orientation of matplotlib
    hm = heatmap!(ax, rotr90(data), colormap = cmap, colorrange = crange)

    # Attach colorbar directly to the right of the specific axis
    if is_categorical
        Colorbar(position[1, 2], hm, ticks = [0, 1], height = Relative(0.7), width = 15)
    else
        Colorbar(position[1, 2], hm, height = Relative(0.7), width = 15)
    end

    return ax
end

# --- Top Row ---
plot_panel!(fig[1, 1], orography_map, "Orography (m)", :terrain, (0, 3500))
plot_panel!(fig[1, 2], lsm_map, "Land-Sea Mask", :grays, (0, 1), is_categorical = true)
plot_panel!(fig[1, 3], vegh_map, "High Veg Fraction (vegh)", :Greens, (0, 1))
plot_panel!(fig[1, 4], vegl_map, "Low Veg Fraction (vegl)", :YlGn, (0, 1))

# --- Bottom Row ---
plot_panel!(fig[2, 1], swl1_3d[:, :, 5], "Soil Moisture L1 (swl1)", :Blues, (0, 1))
plot_panel!(fig[2, 2], swl2_map, "Soil Moisture L2 (swl2)", :Blues, (0, 1))
plot_panel!(fig[2, 3], sst_3d[:, :, 7], "Sea Surface Temp (°C)", :coolwarm, (0, 40))
plot_panel!(fig[2, 4], albedo_map, "Surface Albedo", :bone_1, (0, 0.5))

rowgap!(fig.layout, 10)
colgap!(fig.layout, 10)

display(fig)

# Save output
# save("pangaea_diagnostics.png", fig)
