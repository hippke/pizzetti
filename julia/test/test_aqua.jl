@testset "package quality (Aqua)" begin
    Aqua.test_all(Pizzetti; ambiguities = false, deps_compat = (check_extras = false,))
    Aqua.test_ambiguities(Pizzetti)
end
