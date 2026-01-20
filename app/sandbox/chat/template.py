from e2b import Template

template = (
    Template()
    .from_template("code-interpreter-v1")  # Correct name
    .pip_install([
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "requests",
        "tavily-python",
        "scipy"
    ])
)