









#============== FLO VARIANT ======================================================================================
VARIANT_CONFIG = {
    "variant_id": "FLO",

    "description": "Feature-linear-objective variant. Only objectives and constraints may change.",

    "allowed_override_keys": [
        "paper_objectives",
        "paper_constraints",
    ],

    "overrides": {
        "paper_objectives": {
            "objective_family": "FLO",

            "objectives": [
                # Fill with your actual custom DESC objective class names.
                # Example:
                # {
                #     "class": "FLOPressureAxis",
                #     "kwargs": {
                #         "target": None,
                #         "weight": 1.0,
                #     },
                # },
            ],
        },

        "paper_constraints": [
            # Fill with the exact FLO constraint list.
            # Keep only objective/constraint changes here.
        ],
    },
}
#==============================================================================================================