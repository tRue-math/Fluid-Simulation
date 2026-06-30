using Plots

Base.@kwdef mutable struct CavitySimulation
    Nx::Int; Ny::Int
    Pr::Float64; Ra::Float64 # プラントル数、レイリー数
    dt::Float64; dx::Float64; dy::Float64

    # SOR法の加速緩和係数
    omega::Float64
    
    u::Matrix{Float64}; v::Matrix{Float64}
    u_star::Matrix{Float64}; v_star::Matrix{Float64}

    p::Matrix{Float64}; p_delta::Matrix{Float64} 
    
    T::Matrix{Float64}; T_new::Matrix{Float64}
    
    # ポアソン方程式の右辺（発散）用
    div::Matrix{Float64}
end

function CavitySimulation(; Nx, Ny, Pr, Ra, dt, dx, dy)
    # スタガード格子と仮想セルを考慮した配列の事前確保
    u = zeros(Nx + 1, Ny + 2); v = zeros(Nx + 2, Ny + 1)
    u_star = zeros(Nx + 1, Ny + 2); v_star = zeros(Nx + 2, Ny + 1)
    p = zeros(Nx + 2, Ny + 2); p_delta = zeros(Nx + 2, Ny + 2)
    T = zeros(Nx + 2, Ny + 2); T_new = zeros(Nx + 2, Ny + 2)
    div = zeros(Nx + 2, Ny + 2) # インデックスをpと揃えるために大きめに確保
    
    rho_jacobi = (cos(pi / Nx) * dy^2 + cos(pi / Ny) * dx^2) / (dx^2 + dy^2)
    
    omega_opt = 2.0 / (1.0 + sqrt(1.0 - rho_jacobi^2))

    sim = CavitySimulation(Nx, Ny, Pr, Ra, dt, dx, dy, omega_opt,
                           u, v, u_star, v_star, p, p_delta, T, T_new, div)

    apply_temperature_bc!(sim)
    return sim
end

function step!(sim::CavitySimulation)
    #1. 仮の速度場の更新
    compute_tentative_velocity_interior!(sim)
    apply_velocity_bc!(sim)

    #2. 圧力場の計算
    solve_poisson_sor!(sim)

    #3. 速度補正値の計算
    correct_velocity_and_pressure!(sim)
    apply_velocity_bc!(sim)
    apply_pressure_bc!(sim)

    #4. 温度場の計算
    compute_temperature_interior!(sim)
    apply_temperature_bc!(sim)
end

function compute_tentative_velocity_interior!(sim::CavitySimulation)
    # 変数をローカルにバインド（型推論を助け、メモリアクセスを高速化）
    Nx, Ny = sim.Nx, sim.Ny
    dx, dy, dt = sim.dx, sim.dy, sim.dt
    Pr, Ra = sim.Pr, sim.Ra
    u, v, p, T = sim.u, sim.v, sim.p, sim.T
    u_star, v_star = sim.u_star, sim.v_star

    # ==========================================
    # u* の計算 (内部: i = 2:Nx, j = 2:Ny+1)
    # ==========================================
    @inbounds for j in 2:Ny+1
        for i in 2:Nx
            # スタガード格子なので，uの位置におけるvの値を補間する
            vv = (v[i, j-1] + v[i+1, j-1] + v[i, j] + v[i+1, j]) / 4.0

            # (V･∇)u (移流項) 1次風上差分
            cnvux = u[i,j] >= 0.0 ? u[i,j] * (u[i,j] - u[i-1,j]) / dx :
                                    u[i,j] * (u[i+1,j] - u[i,j]) / dx
            cnvuy = vv >= 0.0     ? vv * (u[i,j] - u[i,j-1]) / dy :
                                    vv * (u[i,j+1] - u[i,j]) / dy

            # 拡散項 2次中心差分
            difu = Pr * ( (u[i-1,j] - 2.0*u[i,j] + u[i+1,j]) / dx^2 +
                          (u[i,j-1] - 2.0*u[i,j] + u[i,j+1]) / dy^2 )

            # 圧力勾配項
            grad_p_x = (p[i+1,j] - p[i,j]) / dx

            # 仮の速度 u* の更新 (浮力項はx方向なので0)
            u_star[i,j] = u[i,j] + dt * (-cnvux - cnvuy + difu - grad_p_x)
        end
    end

    # ==========================================
    # v* の計算 (内部: i = 2:Nx+1, j = 2:Ny)
    # ==========================================
    @inbounds for j in 2:Ny
        for i in 2:Nx+1
            # スタガード格子なので，vの位置におけるuの値を補間する
            uu = (u[i-1, j] + u[i, j] + u[i-1, j+1] + u[i, j+1]) / 4.0
            tv = (T[i, j] + T[i, j+1]) / 2.0

            # (V･∇)v (移流項) 1次風上差分
            cnvvx = uu >= 0.0     ? uu * (v[i,j] - v[i-1,j]) / dx :
                                    uu * (v[i+1,j] - v[i,j]) / dx
            cnvvy = v[i,j] >= 0.0 ? v[i,j] * (v[i,j] - v[i,j-1]) / dy :
                                    v[i,j] * (v[i,j+1] - v[i,j]) / dy

            # 拡散項 2次中心差分
            difv = Pr * ( (v[i-1,j] - 2.0*v[i,j] + v[i+1,j]) / dx^2 +
                          (v[i,j-1] - 2.0*v[i,j] + v[i,j+1]) / dy^2 )

            # 浮力項 ブシネ近似
            buov = (Ra * Pr^2) * tv

            # 圧力勾配項
            grad_p_y = (p[i,j+1] - p[i,j]) / dy

            # 仮の速度 v* の更新
            v_star[i,j] = v[i,j] + dt * (-cnvvx - cnvvy + difv + buov - grad_p_y)
        end
    end
end

function apply_velocity_bc!(sim::CavitySimulation)
    Nx, Ny = sim.Nx, sim.Ny
    u_star, v_star = sim.u_star, sim.v_star

    # ==========================================
    # u_star の境界条件 (サイズ: Nx+1, Ny+2)
    # ==========================================
    @inbounds for j in 1:Ny+2
        # 左右の壁は速度0
        u_star[1, j]    = 0.0
        u_star[Nx+1, j] = 0.0
    end

    @inbounds for i in 1:Nx+1
        # 上下の壁で速度0になるように，仮想セルを設定
        u_star[i, 1]    = -u_star[i, 2]
        u_star[i, Ny+2] = -u_star[i, Ny+1]
    end

    # ==========================================
    # v_star の境界条件 (サイズ: Nx+2, Ny+1)
    # ==========================================
    @inbounds for j in 1:Ny+1
        # 左右の壁で速度0になるように，仮想セルを設定
        v_star[1, j]    = -v_star[2, j]
        v_star[Nx+2, j] = -v_star[Nx+1, j]
    end

    @inbounds for i in 1:Nx+2
        # 上下の壁は速度0
        v_star[i, 1]    = 0.0
        v_star[i, Ny+1] = 0.0
    end
end

function solve_poisson_sor!(sim::CavitySimulation)
    Nx, Ny = sim.Nx, sim.Ny
    dx, dy, dt = sim.dx, sim.dy, sim.dt
    u_star, v_star = sim.u_star, sim.v_star
    p, div = sim.p, sim.div
    p_delta = sim.p_delta
    omega = sim.omega

    fill!(p_delta, 0.0)
    
    # 仮の速度場からdiv(内部)を計算
    @inbounds for j in 2:Ny+1
        for i in 2:Nx+1
            div[i, j] = (u_star[i, j] - u_star[i-1, j]) / dx + 
                        (v_star[i, j] - v_star[i, j-1]) / dy
        end
    end
    
    # 2. SOR法による反秘計算ループ
    max_iter = 500
    eps_p = 1e-4

    idx2 = 1.0 / dx^2
    idy2 = 1.0 / dy^2
    beta = 2.0 * (idx2 + idy2)
    
    for iter in 1:max_iter
        residual = 0.0
        
        # 内部セルの更新 (行列を組まずに、現在の周囲の値から直接更新)
        @inbounds for j in 2:Ny+1
            for i in 2:Nx+1
                p_delta_old = p_delta[i, j]
                
                # 5点差分から導かれる次の値
                p_delta_new = ((p[i+1, j] + p[i-1, j]) * idx2 +
                               (p[i, j+1] + p[i, j-1]) * idy2 - div[i, j] / dt) / beta
                p_delta_new = ((p_delta[i+1, j] + p_delta[i-1, j]) * idx2 +
                               (p_delta[i, j+1] + p_delta[i, j-1]) * idy2 - div[i, j] / dt) / beta
                
                # SOR法の緩和処理
                p_delta[i, j] = p_delta_old + omega * (p_delta_new - p_delta_old)
                
                residual += (p_delta[i, j] - p_delta_old)^2
            end
        end
        
        # 境界条件を適用 (反復のたびに仮想セルを同期させる)
        apply_pressure_bc!(sim)
        
        # 収束判定
        if residual < eps_p
            break
        end
    end
end

function apply_pressure_bc!(sim::CavitySimulation)
    Nx, Ny = sim.Nx, sim.Ny
    p_delta = sim.p_delta
    
    # 左右の壁 (ノイマン条件: 勾配0)
    for j in 1:Ny+2
        p_delta[1, j] = p_delta[2, j]
        p_delta[Nx+2, j] = p_delta[Nx+1, j]
    end
    
    # 上下の壁 (ノイマン条件: 勾配0)
    for i in 1:Nx+2
        p_delta[i, 1] = p_delta[i, 2]
        p_delta[i, Ny+2] = p_delta[i, Ny+1]
    end
    # ここでは基準値を設定しない
end

function apply_temperature_bc!(sim::CavitySimulation)
    Nx, Ny = sim.Nx, sim.Ny
    T = sim.T
    
    # 左右の壁 (ディリクレ条件: 温度固定)
    for j in 1:Ny+2
        # 壁の温度が-0.5,0.5になるように仮想セルを補完
        T[1, j] = -1.0 - T[2, j]
        T[Nx+2, j] = 1.0 - T[Nx+1, j]
    end
    
    # 上下の壁 (フォン・ノイマン条件: 勾配0)
    for i in 1:Nx+2
        T[i, 1] = T[i, 2]
        T[i, Ny+2] = T[i, Ny+1]
    end
end

function correct_velocity_and_pressure!(sim::CavitySimulation)
    Nx, Ny = sim.Nx, sim.Ny
    dx, dy, dt = sim.dx, sim.dy, sim.dt
    u, v = sim.u, sim.v
    u_star, v_star = sim.u_star, sim.v_star
    p, p_delta = sim.p, sim.p_delta

    # ==========================================
    # 1. 速度場 u の修正 (内部セル: i = 2:Nx, j = 2:Ny+1)
    # ==========================================
    @inbounds for j in 2:Ny+1
        for i in 2:Nx
            # u[i,j] は p[i,j] と p[i+1,j] の境界にある
            grad_p_delta_x = (p_delta[i+1, j] - p_delta[i, j]) / dx
            u[i, j] = u_star[i, j] - dt * grad_p_delta_x
        end
    end

    # ==========================================
    # 2. 速度場 v の修正 (内部セル: i = 2:Nx+1, j = 2:Ny)
    # ==========================================
    @inbounds for j in 2:Ny
        for i in 2:Nx+1
            # v[i,j] は p[i,j] と p[i,j+1] の境界にある
            grad_p_delta_y = (p_delta[i, j+1] - p_delta[i, j]) / dy
            v[i, j] = v_star[i, j] - dt * grad_p_delta_y
        end
    end

    # ==========================================
    # 3. 圧力場の更新 (内部セル)
    # ==========================================
    @inbounds for j in 2:Ny+1
        for i in 2:Nx+1
            p[i, j] = p[i, j] + p_delta[i, j]
        end
    end

    # 基準値を設定 (圧力の絶対値は任意なので、左下のセルを0にする)
    p .-= p[2, 2] # 全体を基準値に合わせる
end

function compute_temperature_interior!(sim::CavitySimulation)
    Nx, Ny = sim.Nx, sim.Ny
    dx, dy, dt = sim.dx, sim.dy, sim.dt
    
    u, v, T = sim.u, sim.v, sim.T
    
    # 新しい温度場用の配列を用意(上書きすると参照する値が壊れる)
    T_new = sim.T_new

    @inbounds for j in 2:Ny+1
        for i in 2:Nx+1
            # セルの真ん中のu,vを補完
            uut = (u[i, j] + u[i-1, j]) / 2.0
            vvt = (v[i, j] + v[i, j-1]) / 2.0

            # (V･∇)T (移流項) 1次風上差分
            cnvtx = uut >= 0.0 ? uut * (T[i, j] - T[i-1, j]) / dx : 
                                 uut * (T[i+1, j] - T[i, j]) / dx
            
            cnvty = vvt >= 0.0 ? vvt * (T[i, j] - T[i, j-1]) / dy : 
                                 vvt * (T[i, j+1] - T[i, j]) / dy

            # ∆T 2次中心差分
            # 今回は無次元化でここが1になるように速度の代表値を取っている
            dift = (T[i-1, j] - 2.0*T[i, j] + T[i+1, j]) / dx^2 +
                     (T[i, j-1] - 2.0*T[i, j] + T[i, j+1]) / dy^2

            # 4. 次の時間の温度 T_new を計算
            T_new[i, j] = T[i, j] + dt * (-cnvtx - cnvty + dift)
        end
    end
    
    # 更新
    T .= T_new
end


function main()
    Nx, Ny = 40, 40
    sim = CavitySimulation(Nx=Nx, Ny=Ny, Pr=0.71, Ra=1e5, dt=1e-4, dx=1.0/Nx, dy=1.0/Ny)

    total_steps = 2000
    output_interval = 20

    println("流体シミュレーションを実行中...")
    
    # アニメーション用の空の箱を用意
    anim = Animation()

    for step in 1:total_steps
        step!(sim)

        if step % output_interval == 0
            # 内部セルのみを抽出し、Plots用に転置 (') して渡す
            T_plot = sim.T[2:Nx+1, 2:Ny+1]'
            
            # ヒートマップ描画を変数 plt に保存
            plt = heatmap(1:Nx, 1:Ny, T_plot, 
                    title="Cavity Flow - Step: $step", 
                    c=:thermal,          # カラーマップ (赤〜青)
                    aspect_ratio=:equal, # 縦横比を1:1に
                    xlims=(1, Nx), ylims=(1, Ny),
                    clim=(-0.5, 0.5),    # カラーバーの最小・最大値を固定
                    colorbar_title="Temperature",
                    framestyle=:box)
            
            # ==========================================
            # ★ 速度ベクトル (矢印) の追加プロット
            # ==========================================
            skip = 3      # 矢印を描く間隔 (全セルに描くと真っ黒になるので間引く)
            scale = 0.05  # 矢印の長さを調整するスケール係数
            
            xs, ys, us, vs = Float64[], Float64[], Float64[], Float64[]
            
            for i in 1:skip:Nx
                for j in 1:skip:Ny
                    # スタガード格子なので、セル中心の速度を周囲の境界から補間して求める
                    # 内部セルのインデックスは[i+1, j+1]に対応します
                    uc = (sim.u[i, j+1] + sim.u[i+1, j+1]) / 2.0
                    vc = (sim.v[i+1, j] + sim.v[i+1, j+1]) / 2.0
                    
                    # 速度がほぼ0の場所は矢印を描かないようにする (見栄えのため)
                    if sqrt(uc^2 + vc^2) > 1e-3
                        push!(xs, i)
                        push!(ys, j)
                        push!(us, uc * scale)
                        push!(vs, vc * scale)
                    end
                end
            end
            
            # ヒートマップの上に白い矢印を重ねて描画する (quiver!)
            quiver!(plt, xs, ys, quiver=(us, vs), color=:white, linewidth=1.0)
            
            # 描画したグラフをフレームとしてアニメーションに追加
            frame(anim, plt)
        end
    end

    # GIFアニメーションとして保存
    gif(anim, "cavity_flow.gif", fps=15)
    println("✅ cavity_flow.gif の生成が完了しました！")
end

# 実行
main()