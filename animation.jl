using Printf
using Plots
using DelimitedFiles

function read_csv_matrix(path::AbstractString)
	if filesize(path) == 0
		return zeros(Float64, 0, 0)
	end
	data = readdlm(path, ',')
	if isempty(data)
		return zeros(Float64, 0, 0)
	end
	return Matrix{Float64}(data)
end

function collect_step_files(output_dir::AbstractString, prefix::AbstractString)
	pattern = Regex("^" * prefix * "_step_(\\d+)\\.csv\$")
	step_to_path = Dict{Int, String}()

	for name in readdir(output_dir)
		match_result = match(pattern, name)
		if match_result !== nothing
			step = parse(Int, match_result.captures[1])
			step_to_path[step] = joinpath(output_dir, name)
		end
	end

	steps = sort!(collect(keys(step_to_path)))
	return steps, step_to_path
end

function sample_velocity_vectors(data::Matrix{Float64})
	if isempty(data)
		return Float64[], Float64[], Float64[], Float64[]
	end

	xs = Float64[]
	ys = Float64[]
	us = Float64[]
	vs = Float64[]

	for row in 1:size(data, 1)
		x = data[row, 1]
		y = data[row, 2]
		u = data[row, 3]
		v = data[row, 4]

		if u^2 + v^2 > 1e-12
			push!(xs, x)
			push!(ys, y)
			push!(us, u)
			push!(vs, v)
		end
	end

	return xs, ys, us, vs
end

function build_gifs_from_csv(output_dir::AbstractString="output_data"; gif_dir::AbstractString=".", fps::Int=15)
	isdir(output_dir) || error("CSV ディレクトリが見つかりません: $(output_dir)")
	mkpath(gif_dir)

	t_steps, t_files = collect_step_files(output_dir, "T")
	p_steps, p_files = collect_step_files(output_dir, "p")
	omega_steps, omega_files = collect_step_files(output_dir, "omega")
	velocity_steps, velocity_files = collect_step_files(output_dir, "velocity")

	if isempty(omega_steps)
		error("omega_step_*.csv が見つかりません。main_hpc の出力先を確認してください。")
	end

	common_steps = t_steps
	if isempty(common_steps)
		common_steps = p_steps
	end
	if isempty(common_steps)
		common_steps = omega_steps
	end

	if isempty(common_steps)
		error("アニメーション化できる CSV が見つかりません。")
	end

	anim_T = Animation()
	anim_p = Animation()
	anim_v = Animation()
	anim_omega = Animation()

	first_omega = read_csv_matrix(omega_files[first(common_steps)])
	Ny, Nx = size(first_omega)

	for step in common_steps
		if haskey(t_files, step)
			T_plot = read_csv_matrix(t_files[step])
			plt_T = heatmap(1:Nx, 1:Ny, T_plot,
				title=@sprintf("Temperature - Step: %06d", step),
				c=:thermal,
				aspect_ratio=:equal,
				xlims=(1, Nx), ylims=(1, Ny),
				clim=(-0.5, 0.5),
				colorbar_title="T",
				framestyle=:box)

			if haskey(velocity_files, step)
				velocity_data = read_csv_matrix(velocity_files[step])
				xs, ys, us, vs = sample_velocity_vectors(velocity_data)
				quiver!(plt_T, xs, ys, quiver=(us, vs), color=:white, linewidth=1.0)
			end
			frame(anim_T, plt_T)
		end

		if haskey(p_files, step)
			p_plot = read_csv_matrix(p_files[step])
			plt_p = heatmap(1:Nx, 1:Ny, p_plot,
				title=@sprintf("Pressure - Step: %06d", step),
				c=:viridis,
				aspect_ratio=:equal,
				xlims=(1, Nx), ylims=(1, Ny),
				colorbar_title="P",
				framestyle=:box)

			if haskey(velocity_files, step)
				velocity_data = read_csv_matrix(velocity_files[step])
				xs, ys, us, vs = sample_velocity_vectors(velocity_data)
				quiver!(plt_p, xs, ys, quiver=(us, vs), color=:white, linewidth=1.0)
			end
			frame(anim_p, plt_p)
		end

		if haskey(velocity_files, step)
			velocity_data = read_csv_matrix(velocity_files[step])
			xs, ys, us, vs = sample_velocity_vectors(velocity_data)

			plt_v = plot(title=@sprintf("Velocity Vectors - Step: %06d", step),
				aspect_ratio=:equal,
				xlims=(1, Nx), ylims=(1, Ny),
				framestyle=:box,
				legend=false,
				background_color=:white)

			quiver!(plt_v, xs, ys, quiver=(us, vs), color=:black, linewidth=1.2)
			frame(anim_v, plt_v)
		end

		if haskey(omega_files, step)
			omega_plot = read_csv_matrix(omega_files[step])
			omega_max = max(maximum(abs, omega_plot), 1e-12)

			plt_omega = heatmap(1:Nx, 1:Ny, omega_plot,
				title=@sprintf("Vorticity - Step: %06d", step),
				c=:balance,
				aspect_ratio=:equal,
				xlims=(1, Nx), ylims=(1, Ny),
				clim=(-omega_max, omega_max),
				colorbar_title="ω",
				framestyle=:box)

			if haskey(velocity_files, step)
				velocity_data = read_csv_matrix(velocity_files[step])
				xs, ys, us, vs = sample_velocity_vectors(velocity_data)
				quiver!(plt_omega, xs, ys, quiver=(us, vs), color=:white, linewidth=1.0)
			end
			frame(anim_omega, plt_omega)
		end
	end

	gif_path_T = joinpath(gif_dir, "cavity_flow_T.gif")
	gif_path_p = joinpath(gif_dir, "cavity_flow_p.gif")
	gif_path_v = joinpath(gif_dir, "cavity_flow_velocity.gif")
	gif_path_omega = joinpath(gif_dir, "cavity_flow_vorticity.gif")

	if !isempty(t_steps)
		gif(anim_T, gif_path_T, fps=fps)
		println("saved: ", gif_path_T)
	end
	if !isempty(p_steps)
		gif(anim_p, gif_path_p, fps=fps)
		println("saved: ", gif_path_p)
	end
	if !isempty(velocity_steps)
		gif(anim_v, gif_path_v, fps=fps)
		println("saved: ", gif_path_v)
	end
	if !isempty(omega_steps)
		gif(anim_omega, gif_path_omega, fps=fps)
		println("saved: ", gif_path_omega)
	end

	return nothing
end

function main()
	build_gifs_from_csv("output_data"; gif_dir=".", fps=15)
end

main()
