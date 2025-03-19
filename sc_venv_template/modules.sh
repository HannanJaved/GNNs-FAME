module purge
module load GCC OpenMPI
# Some base modules commonly used in AI
module load tqdm matplotlib IPython bokeh git
module load Seaborn OpenCV

# ML Frameworks
module load PyTorch scikit-learn torchvision PyTorch-Lightning
module load tensorboard