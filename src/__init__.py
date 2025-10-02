import dolfin as fe

# FEniCS settings
fe.parameters["form_compiler"]["cpp_optimize"] = True
fe.parameters["form_compiler"]["representation"] = "uflacs"
fe.parameters["linear_algebra_backend"] = "PETSc"
