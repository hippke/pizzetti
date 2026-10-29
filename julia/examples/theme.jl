# Shared style of the examples: Computer Modern type (also used by GR's LaTeX labels) and the
# colours of the paper's anatomy figure (interior series in blue, limb rule in orange) for
# Plots.jl, and `table` for results.

using Plots, LaTeXStrings, Markdown, Printf
ENV["GKSwstype"] = "100"                     # headless GR, for scripts and notebook builds

const COLORS = (series = colorant"#2f6aa3",  # interior series S_M
                limb = colorant"#d4601f",    # limb rule (trapezoid / interpolants)
                quadratic = colorant"#5f8f3a",
                contact = colorant"#7a5aa6",
                quadrature = colorant"#8a8a3c",
                out = colorant"#707070",
                ink = colorant"#262626",
                accent = colorant"#a31f34")
const FILLS = (series = colorant"#b7d0e8", limb = colorant"#f6c6ae", quadrature = colorant"#dcdca8",
               out = colorant"#f0f0f0")
const METHOD_COLORS = Dict(:series => COLORS.series, :interpolate => COLORS.limb, :trapezoid => COLORS.limb,
                           :quadratic => COLORS.quadratic, :contact => COLORS.contact, :proven => COLORS.contact,
                           :quadrature => COLORS.quadrature, :out => COLORS.out, :full => COLORS.out)

default(fontfamily = "Computer Modern", titlefontsize = 11, guidefontsize = 11, tickfontsize = 9,
        legendfontsize = 9, legend_foreground_color = nothing, legend_background_color = RGBA(1, 1, 1, 0.8),
        framestyle = :box, grid = false, minorticks = false, linewidth = 1.8, dpi = 150,
        size = (720, 400), left_margin = 3Plots.mm, bottom_margin = 3Plots.mm,
        palette = [COLORS.series, COLORS.limb, COLORS.quadratic, COLORS.contact, COLORS.accent, COLORS.out])

"""
    table(header, rows; align = "lrr…")

A table of results: HTML in notebooks, Markdown in a terminal. `rows` is a collection of tuples
or vectors; strings are shown as they are (inline code in backticks), other values with
`string`. `align` has one letter per column, `l` or `r`.
"""
struct Table
    header::Vector{String}
    rows::Vector{Vector{String}}
    align::String
end
table(header, rows; align = "l" * "r"^(length(header) - 1)) =
    Table(string.(header), [[x isa AbstractString ? x : string(x) for x in r] for r in rows], align)

function Base.show(io::IO, ::MIME"text/plain", t::Table)
    rule = join([a == 'l' ? ":---" : "---:" for a in t.align], "|")
    lines = ["| " * join(t.header, " | ") * " |", "|" * rule * "|"]
    append!(lines, ["| " * join(r, " | ") * " |" for r in t.rows])
    print(io, join(lines, "\n"))
end
function Base.show(io::IO, ::MIME"text/html", t::Table)
    esc(s) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")
    code(s) = replace(esc(s), r"`([^`]*)`" => s"<code>\1</code>")
    al(j) = t.align[j] == 'l' ? "left" : "right"
    print(io, "<table>\n<thead><tr>")
    for (j, h) in enumerate(t.header)
        print(io, "<th style=\"text-align:", al(j), "\">", code(h), "</th>")
    end
    print(io, "</tr></thead>\n<tbody>\n")
    for r in t.rows
        print(io, "<tr>")
        for (j, x) in enumerate(r)
            print(io, "<td style=\"text-align:", al(j), "\">", code(x), "</td>")
        end
        print(io, "</tr>\n")
    end
    print(io, "</tbody>\n</table>")
end

"Format a number in scientific notation with `d` significant digits, e.g. `sci(3.2e-15) == \"3.2e-15\"`."
sci(x, d = 2) = x == 0 ? "0" : @sprintf("%.*e", d - 1, x)
nothing
