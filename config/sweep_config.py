sweep_config = {
    'method': 'grid',  
    'metric': {
        'goal': 'minimize',  
        'name': 'rmse' 
    },
    'parameters': {
        'model': {
            'values': ['GradientBoosting']
        },
        'n_estimators': {
            'values': [100, 200]
        },
        'learning_rate': {
            'values': [0.01, 0.1]
        },
        'max_depth': {
            'values': [10, 15, 20]
        },
    },
}
