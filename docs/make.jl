using WVTICs
using Documenter

DocMeta.setdocmeta!(WVTICs, :DocTestSetup, :(using WVTICs); recursive=true)

makedocs(;
    modules=[WVTICs],
    authors="Ludwig Böss",
    sitename="WVTICs.jl",
    format=Documenter.HTML(;
        canonical="https://LudwigBoess.github.io/WVTICs.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Usage" => "usage.md",
        "Problems" => "problems.md",
        "Parallelism" => "parallel.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/LudwigBoess/WVTICs.jl",
    devbranch="main",
)
