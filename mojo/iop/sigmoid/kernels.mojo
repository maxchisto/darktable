from std.math import isnan, sqrt, pow, max, min

@always_inline
def generalized_loglogistic_sigmoid_scalar(
    value: Float32,
    magnitude: Float32,
    paper_exp: Float32,
    film_fog: Float32,
    film_power: Float32,
    paper_power: Float32,
) -> Float32:
    var clamped_value = max(value, Float32(0.0))
    var film_response = pow(film_fog + clamped_value, film_power)
    var paper_response = magnitude * pow(
        film_response / (paper_exp + film_response), paper_power
    )
    if isnan(paper_response):
        return magnitude
    return paper_response

@always_inline
def apply_sigmoid_rgb_ratio(
    in_r: Float32,
    in_g: Float32,
    in_b: Float32,
    in_a: Float32,
    white_target: Float32,
    black_target: Float32,
    paper_exp: Float32,
    film_fog: Float32,
    film_power: Float32,
    paper_power: Float32,
) -> SIMD[DType.float32, 4]:
    var avg = max((in_r + in_g + in_b) / Float32(3.0), Float32(0.0))
    var min_v = min(min(in_r, in_g), in_b)
    var sat = Float32(1.0)
    if min_v < Float32(0.0):
        sat = -avg / (min_v - avg)

    var p_r = avg + sat * (in_r - avg)
    var p_g = avg + sat * (in_g - avg)
    var p_b = avg + sat * (in_b - avg)

    var luma = (p_r + p_g + p_b) / Float32(3.0)
    var mapped_luma = generalized_loglogistic_sigmoid_scalar(
        luma, white_target, paper_exp, film_fog, film_power, paper_power
    )

    if luma > Float32(1e-9):
        var scale = mapped_luma / luma
        p_r *= scale
        p_g *= scale
        p_b *= scale
    else:
        p_r = mapped_luma
        p_g = mapped_luma
        p_b = mapped_luma

    var p_min = min(min(p_r, p_g), p_b)
    var p_max = max(max(p_r, p_g), p_b)
    var eps = Float32(1e-6)
    var d_white = (white_target - mapped_luma) / (p_max - mapped_luma + eps)
    var d_black = (black_target - mapped_luma) / (p_min - mapped_luma - eps)
    var db_vs_chroma = min(d_white, d_black)
    var cvm_border = (mapped_luma - p_min) / (mapped_luma + eps)
    var p_chr_adj = Float32(1.0) / (cvm_border * db_vs_chroma + eps)
    var h_chr = (Float32(2.0) * cvm_border / (Float32(1.0) - cvm_border * cvm_border + eps)) * p_chr_adj
    var h_z = sqrt(h_chr * h_chr + Float32(1.0))
    var chroma_f = h_chr / (Float32(1.0) + h_z) * db_vs_chroma

    return SIMD[DType.float32, 4](
        mapped_luma + chroma_f * (p_r - mapped_luma),
        mapped_luma + chroma_f * (p_g - mapped_luma),
        mapped_luma + chroma_f * (p_b - mapped_luma),
        in_a,
    )

@always_inline
def apply_sigmoid_per_channel(
    in_r: Float32,
    in_g: Float32,
    in_b: Float32,
    in_a: Float32,
    white_target: Float32,
    paper_exp: Float32,
    film_fog: Float32,
    contrast_power: Float32,
    skew_power: Float32,
    hue_preservation: Float32,
    pipe_to_base: SIMD[DType.float32, 16],
    base_to_rendering: SIMD[DType.float32, 16],
    rendering_to_pipe: SIMD[DType.float32, 16],
) -> SIMD[DType.float32, 4]:
    var i_r = pipe_to_base[0] * in_r + pipe_to_base[1] * in_g + pipe_to_base[2] * in_b
    var i_g = pipe_to_base[4] * in_r + pipe_to_base[5] * in_g + pipe_to_base[6] * in_b
    var i_b = pipe_to_base[8] * in_r + pipe_to_base[9] * in_g + pipe_to_base[10] * in_b

    var avg = max((i_r + i_g + i_b) / Float32(3.0), Float32(0.0))
    var min_v = min(min(i_r, i_g), i_b)
    var sat = Float32(1.0)
    if min_v < Float32(0.0):
        sat = -avg / (min_v - avg)
    i_r = avg + sat * (i_r - avg)
    i_g = avg + sat * (i_g - avg)
    i_b = avg + sat * (i_b - avg)

    var r_r = base_to_rendering[0] * i_r + base_to_rendering[1] * i_g + base_to_rendering[2] * i_b
    var r_g = base_to_rendering[4] * i_r + base_to_rendering[5] * i_g + base_to_rendering[6] * i_b
    var r_b = base_to_rendering[8] * i_r + base_to_rendering[9] * i_g + base_to_rendering[10] * i_b

    var pc_r = generalized_loglogistic_sigmoid_scalar(
        r_r, white_target, paper_exp, film_fog, contrast_power, skew_power
    )
    var pc_g = generalized_loglogistic_sigmoid_scalar(
        r_g, white_target, paper_exp, film_fog, contrast_power, skew_power
    )
    var pc_b = generalized_loglogistic_sigmoid_scalar(
        r_b, white_target, paper_exp, film_fog, contrast_power, skew_power
    )

    var p_min: Float32
    var p_mid: Float32
    var p_max: Float32
    var pc_min: Float32
    var pc_mid: Float32
    var pc_max: Float32

    if r_r >= r_g:
        if r_g >= r_b:
            p_max = r_r; p_mid = r_g; p_min = r_b
            pc_max = pc_r; pc_mid = pc_g; pc_min = pc_b
        elif r_b >= r_r:
            p_max = r_b; p_mid = r_r; p_min = r_g
            pc_max = pc_b; pc_mid = pc_r; pc_min = pc_g
        else:
            p_max = r_r; p_mid = r_b; p_min = r_g
            pc_max = pc_r; pc_mid = pc_b; pc_min = pc_g
    else:
        if r_r >= r_b:
            p_max = r_g; p_mid = r_r; p_min = r_b
            pc_max = pc_g; pc_mid = pc_r; pc_min = pc_b
        elif r_b >= r_g:
            p_max = r_b; p_mid = r_g; p_min = r_r
            pc_max = pc_b; pc_mid = pc_g; pc_min = pc_r
        else:
            p_max = r_g; p_mid = r_b; p_min = r_r
            pc_max = pc_g; pc_mid = pc_b; pc_min = pc_r

    var chroma = p_max - p_min
    var midscale = Float32(0.0)
    if chroma != Float32(0.0):
        midscale = (p_mid - p_min) / chroma

    var f_hc = pc_min + (pc_max - pc_min) * midscale
    var n_mid = (Float32(1.0) - hue_preservation) * pc_mid + hue_preservation * f_hc

    var blend = Float32(2.0) * p_min / (p_min + p_mid + Float32(1e-9))
    var target = blend * (pc_r + pc_g + pc_b) + (Float32(1.0) - blend) * (
        pc_min + n_mid + pc_max
    )

    var res_min: Float32
    var res_mid: Float32
    var res_max: Float32
    if n_mid <= pc_mid:
        res_mid = (
            (Float32(1.0) - hue_preservation) * pc_mid
            + hue_preservation
            * (midscale * pc_max + (Float32(1.0) - midscale) * (target - pc_max))
        ) / (Float32(1.0) + hue_preservation * (Float32(1.0) - midscale))
        res_min = target - pc_max - res_mid
        res_max = pc_max
    else:
        res_mid = (
            (Float32(1.0) - hue_preservation) * pc_mid
            + hue_preservation
            * (pc_min * (Float32(1.0) - midscale) + midscale * (target - pc_min))
        ) / (Float32(1.0) + hue_preservation * midscale)
        res_min = pc_min
        res_max = target - pc_min - res_mid

    var res_r: Float32
    var res_g: Float32
    var res_b: Float32
    if r_r >= r_g:
        if r_g >= r_b:
            res_r = res_max; res_g = res_mid; res_b = res_min
        elif r_b >= r_r:
            res_b = res_max; res_r = res_mid; res_g = res_min
        else:
            res_r = res_max; res_b = res_mid; res_g = res_min
    else:
        if r_r >= r_b:
            res_g = res_max; res_r = res_mid; res_b = res_min
        elif r_b >= r_g:
            res_b = res_max; res_g = res_mid; res_r = res_min
        else:
            res_g = res_max; res_b = res_mid; res_r = res_min

    var out_r = rendering_to_pipe[0] * res_r + rendering_to_pipe[1] * res_g + rendering_to_pipe[2] * res_b
    var out_g = rendering_to_pipe[4] * res_r + rendering_to_pipe[5] * res_g + rendering_to_pipe[6] * res_b
    var out_b = rendering_to_pipe[8] * res_r + rendering_to_pipe[9] * res_g + rendering_to_pipe[10] * res_b

    return SIMD[DType.float32, 4](out_r, out_g, out_b, in_a)
