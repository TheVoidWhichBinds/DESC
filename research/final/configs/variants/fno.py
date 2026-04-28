









#============== FNO VARIANT ======================================================================================
VARIANT_CONFIG = {
    "variant_id": "FNO",

    "description": "Feature-nonlinear-objective variant. Only objectives and constraints may change.",

    "allowed_override_keys": [
        "paper_objectives",
        "paper_constraints",
    ],

    "overrides": {
        "paper_objectives": {
            "objective_family": "FNO",

            "objectives": [
                # Fill with your actual custom DESC objective class names.
            ],
        },

        "paper_constraints": [
            # Fill with the exact FNO constraint list.
        ],
    },
}
#==============================================================================================================