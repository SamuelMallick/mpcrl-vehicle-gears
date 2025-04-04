solver_options = {
    "ipopt": {
        "expand": True,
        "show_eval_warnings": True,
        "warn_initial_bounds": True,
        "print_time": False,
        "record_time": True,
        "bound_consistency": True,
        "calc_lam_x": True,
        "calc_lam_p": False,
        "ipopt": {
            "max_iter": 500,
            "sb": "yes",
            "print_level": 0,
        },
    },
    "qrqp": {
        "expand": True,
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
        "print_info": False,
        "print_iter": False,
        "print_header": False,
        "max_iter": 2000,
    },
    "qpoases": {
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
        "printLevel": "none",
        "jit": True,
    },
    "gurobi": {
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
        "gurobi": {
            "OutputFlag": 0,
            "LogToConsole": 0,
        },
    },
    "bonmin": {
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
        "bonmin": {
            "print_level": 0,
            "max_iter": 1000,
        },
    },
    "knitro": {
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
        "knitro": {
            "outlev": 0,
            "maxit": 1000,
        },
    },
    "clp": {
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
    },
}
