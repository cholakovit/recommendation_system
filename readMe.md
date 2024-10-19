watchmedo auto-restart --patterns="*.py" --recursive python3 recommendation_system.py

recommendation_system/
├── data_loader.py      # Contains dataset loading and processing class
├── model.py            # Contains the recommendation model class
├── trainer.py          # Contains the training and evaluation logic in a class
├── recommender.py      # Contains the recommendation logic in a class
├── utils.py            # The main file that orchestrates everything
├── main.py             # Utility functions (like saving/loading the model)
├── user_ratings.csv    # Dataset (example file)


"Welcome to this tutorial where I’ll walk you through building a recommendation system using PyTorch. In this project, we'll create a collaborative filtering model that learns from user-item interaction data, like movie ratings or product preferences, and makes personalized recommendations. We’ve structured the application for simplicity and modularity, separating data loading, model building, training, and prediction into different components. By the end of this video, you’ll understand how to design and implement a powerful recommendation engine using machine learning techniques. Let’s get started!"


in the terminal:
PS C:\Users\spasv\OneDrive\Работен плот\WEB\python\LLM\recommendation_system> python -m venv env 
PS C:\Users\spasv\OneDrive\Работен плот\WEB\python\LLM\recommendation_system> pip install torch pandas
PS C:\Users\spasv\OneDrive\Работен плот\WEB\python\LLM\recommendation_system> python -m pip install --upgrade pip
PS C:\Users\spasv\OneDrive\Работен плот\WEB\python\LLM\recommendation_system> python main.py


A virtual environment is an isolated Python environment. It allows you to manage project-specific dependencies separately from the system-wide Python packages. This is particularly useful when working on multiple projects with potentially conflicting package versions or dependencies.

Benefits of Using Virtual Environments:
- Dependency Isolation: Each virtual environment has its own installed packages, so changes in one environment won't affect others.
- Avoiding Conflicts: You can work with different versions of the same library across different projects without conflicts.
- Easy Reproducibility: By keeping dependencies isolated, it's easier to reproduce the environment for a project on another system.

This command downloads and installs the PyTorch package (and any required dependencies) from the Python Package Index (PyPI) into your current Python environment.

The command python -m pip install is used to run the pip package installer within the current Python environment, allowing you to install Python packages.




