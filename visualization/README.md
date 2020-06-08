# Add visualization 
## The visualization tools include:
+ Visualize results across methods;
+ Visualize results across #tasks;


## Visualization preparation:
+ How to use visualization modules?
    - Default  
    Present both across methods with spider plot and bar plot and across #tasks with line plot
    - Only across methods plot  
    `
    main.py --vis-cross-tasks
    `
    - Only across #task plot  
    `
    main.py --vis-cross-methods
    `
- Arguments
    * Performance matrix:  
    `
    main.py --matrices=ACC,BWT,FWT,Overall ACC
    `
    * Continual learning methods:  
    depends on how many methods selected  
    e.g. `main.py --si --ewc`   
    then `method_name = ['SI', 'EWC']`
    * Results for cross methods:  
    2D array with rows indicating matrices, columns indicating methods.  
    * Results for cross #tasks:  
    **n-runs**: 3D array with rows indicating methods, columns indicating tasks, depth#1 indicating avg., depth#2 indicating std.   
    **1-run**: 2D array with rows indicating methods, columns indicating tasks  
- How to choose the plot type for across methods results?
    * Default:  
    present both spider plot and bar plot
    * Only spider plot:  
    `
    main.py --vis-cross-methods-type=spider
    `
    * Only bar plot:  
    `
    main.py --vis-cross-methods-type=bar
    `