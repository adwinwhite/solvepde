using SparseArrays
using LinearAlgebra
using SpecialFunctions
using DSP
using FFTW
using Plots
using Arpack

const t_n = -2.7

function get_random_state(size::Int64)
    state = rand(size, 1)
    total = sum([abs(c)^2 for c in state])
    return state / sqrt(total)
end

function is_A_site(i::Int64, j::Int64)
    if (j % 2 == 0 && i % 2 == 1) || (j % 2 == 1 && i % 2 == 0)
        return true
    end
    return false
end

function flatten_row_hamiltonian(i::Int64, j::Int64, i_size::Int64, j_size::Int64)
    println(i, ", ", j)
    rowH = zeros(Float64, j_size, i_size)
    if is_A_site(i, j)
        rowH[j, i == i_size ? 1 : i + 1] = -t_n
        rowH[j, i == 1 ? i_size : i - 1] = -t_n
        rowH[j == 1 ? j_size : j - 1, i] = -t_n
    else
        rowH[j, i == i_size ? 1 : i + 1] = -t_n
        rowH[j, i == 1 ? i_size : i - 1] = -t_n
        rowH[j == j_size ? 1 : j + 1, i] = -t_n
    end
    return sparse(reshape(rowH, j_size * i_size, 1))
end

function get_hamiltonian(horizontal_size::Int64, vertical_size::Int64)
    return hcat([flatten_row_hamiltonian(i, j, horizontal_size, vertical_size) for j in 1:vertical_size for i in 1:horizontal_size]...)
end

function get_evolution_operator_onestep_forward(horizontal_size::Int64, vertical_size::Int64, timestep::Float64, allowed_error::Float64=10^(-13), max_order_of_chebyshev_poly::Int64=10000)
    H = get_hamiltonian(horizontal_size, vertical_size)
    #  λ1, ϕ1 = eigs(H, nev=1, which=:LM)
    #  max_eigenvalue = λ1[1]
    max_eigenvalue = 10
    #  λ2, ϕ2 = eigs(H, nev=1, which=:SM)
    #  min_eigenvalue = λ2[1]
    min_eigenvalue = -10

    operator_size = horizontal_size * vertical_size
    z = (max_eigenvalue - min_eigenvalue) * timestep / 2
    B = ((H .- sparse(1.0I, operator_size, operator_size) * (max_eigenvalue + min_eigenvalue) / 2.0) / (max_eigenvalue - min_eigenvalue)) * (-1im) * 2

    evolution_operator = spzeros(operator_size, operator_size)
    T_tilde1 = sparse(1.0I, operator_size, operator_size)
    T_tilde2 = B
    jv = 1
    i = 1
    while abs(jv) > allowed_error && i <= max_order_of_chebyshev_poly
        jv = besselj(i, z)
        evolution_operator = evolution_operator .+ jv * T_tilde2

        next_T_tilde = B * 2 * T_tilde2 .- T_tilde1
        T_tilde1 = T_tilde2
        T_tilde2 = next_T_tilde
        i += 1
        println(i)
    end
    evolution_operator = (evolution_operator * 2 .+ sparse(1.0I, operator_size, operator_size) * besselj(0, z)) * exp((max_eigenvalue + min_eigenvalue) * timestep * (-0.5im))
    return evolution_operator
end

function normalize_state(state)
    total = sum([abs(c)^2 for c in state])
    return state / sqrt(total)
end

function get_correlation(horizontal_size::Int64, vertical_size::Int64, timestep::Float64, sample_size::Int64)
    initial_state = get_random_state(horizontal_size * vertical_size)
    current_state = deepcopy(initial_state)
    evolution_operator = get_evolution_operator_onestep_forward(horizontal_size, vertical_size, timestep)
    correlations= Array{Complex{Float64}}(undef, sample_size)
    for i in 1:sample_size
        current_state = normalize_state(evolution_operator * current_state)
        correlations[i] = dot(initial_state, current_state)
        println(i)
    end
    return correlations
end

function get_dos(correlations::Array{Complex{Float64}, 1}, timestep::Float64)
    #  windowed_corr = correlations .* Windows.hanning(length(correlations), padding=floor(Int64, 0.05 * length(correlations)), zerophase=true)
    windowed_corr = correlations .* Windows.hanning(length(correlations), padding=0, zerophase=true)
    g = fft(windowed_corr) * timestep
    w = [0:(length(correlations)/2-1); (-length(correlations)/2):-1] / length(correlations) * 2 * π / timestep / abs(t_n)
    return w, g
end

function draw_dos(horizontal_size::Int64, vertical_size::Int64, timestep::Float64, sample_size::Int64)
    # horizontal_size = convert(Int64, ARGS[1])
    # vertical_size = convert(Int64, ARGS[2])
    # timestep = convert(Float64, ARGS[3])
    # const
    corrs = get_correlation(horizontal_size, vertical_size, timestep, sample_size)
    w, g = get_dos(corrs, timestep)
    display(plot(w, abs.(g), seriestype = :scatter))
end
