# Docker Experiments

Build Docker image with current ACS implementation and Jupyter Lab image   

    docker build -f Dockerfile.experiments -t acs . 
    
You can run it locally to test if everything is working fine

    docker run --rm --name jupyter -p 9999:9999 -v `pwd`/notebooks:/code/notebooks acs

Finally push to docker hub
    
    docker login
    docker tag acs acs/notebook
    docker push acs/notebook

## Running experiments with Papermill

To run Jupyter notebook to see processed results type:

    docker run -d -v `pwd`/notebooks:/code/notebooks --name jupyter -p 9999:9999 khozzy/acs-notebook

To run a single parametrized experiment type something like:

    docker run -d --rm -v `pwd`/notebooks:/code/notebooks --name mountaincar khozzy/acs-notebook papermill "notebooks/ACS2_in_MountainCar.ipynb" "notebooks/mc_500k_DecayFalse_10bins.ipynb" --log-output -p trials 500000 -p decay False -p bins 10
    
    docker run --rm -v `pwd`/notebooks:/code/notebooks --name "MC" acs:3 papermill "notebooks/MountainCar.ipynb" "notebooks/results/mc.ipynb" --log-output -p trials 1000 -p decay False -p bins 14 -p gamma 0.97 -p biased_exploration 0


