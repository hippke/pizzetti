# Build the executed Jupyter notebooks J*.ipynb from the Literate.jl scripts J*.jl of this folder.
#
#     julia --project=. make_notebooks.jl            # all notebooks
#     julia --project=. make_notebooks.jl J1 J6      # a subset
#
# Each notebook is built in a fresh Julia process, so that it starts from the same state as a
# reader's kernel. The scripts stay the single source: lines ending in `#nb` appear only in the
# notebooks, lines ending in `#jl` only in the scripts (see the Literate.jl documentation).

const HERE = @__DIR__

if isempty(ARGS) || ARGS[1] != "--one"
    names = sort(filter(f -> occursin(r"^J\d_.*\.jl$", f), readdir(HERE)))
    isempty(ARGS) || (names = filter(f -> any(a -> startswith(f, a), ARGS), names))
    for name in names
        t = @elapsed run(`$(Base.julia_cmd()) --project=$HERE $(@__FILE__) --one $name`)
        println(rpad(name, 32), round(t, digits = 1), " s")
    end
    exit()
end

using Literate, JSON, Markdown

# Notebook outputs: keep one PNG per figure (GR also offers SVG and an HTML copy, which triple
# the size), give Markdown results (tables) an HTML form, which GitHub renders, and replace
# local paths by paths relative to the repository.
const ROOT = rstrip(normpath(joinpath(HERE, "..", "..")), '/')
function clean_text(s::AbstractString)
    for r in (ROOT, Base.contractuser(ROOT))
        s = replace(s, r => "pizzetti")
    end
    s = replace(s, homedir() => "~")
    return s
end
clean(x) = x
clean(x::AbstractString) = clean_text(x)
clean(x::AbstractVector) = map(clean, x)
function clean(d::AbstractDict)
    haskey(d, "image/png") && (delete!(d, "image/svg+xml"); delete!(d, "text/html"))
    if haskey(d, "text/markdown") && !haskey(d, "text/html")
        d["text/html"] = Markdown.html(Markdown.parse(join(d["text/markdown"])))
    end
    for (k, v) in d
        d[k] = k == "image/png" ? v : clean(v)
    end
    return d
end

name = ARGS[2]
nbpath = Literate.notebook(joinpath(HERE, name), HERE; execute = true, credit = false)
nb = JSON.parsefile(nbpath)
for cell in nb["cells"]
    cell["cell_type"] == "code" || continue
    cell["outputs"] = [clean(o) for o in cell["outputs"]]
end
nb["metadata"]["kernelspec"] = Dict("display_name" => "Julia $(VERSION.major).$(VERSION.minor)",
                                    "language" => "julia", "name" => "julia-$(VERSION.major).$(VERSION.minor)")
open(nbpath, "w") do io
    JSON.print(io, nb, 1)
end
