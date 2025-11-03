
We will be using the ARM64 cluster in CCR. To check the available nodes in that cluster use the following command

```bash
snodes all ub-hpc/arm64 | grep gpu
```

To create a new job, there are two ways: Using salloc or sbatch. 

### Use this if you want to attach a terminal 

Request for allocation:
```bash
 salloc     --partition=arm64     --qos=arm64     --mem=60G     --nodes=1     --time=3:00:00     --ntasks-per-node=1     --cpus-per-task=32     --gres=gpu:1     --no-shell
```

- `mem` parameter can be increased to 100G or even more, it supports up to 120G or 400G. Will have to confirm.
- `cpus-per-task` can go up to 72
- `time` maximum you can ask for is 72 hours 

Rest keep everything the same, there is a possibility to request 2 nodes at once for parallel training using Pytorch Lightning but I have not been able to make that work

Once your job request is granted, you will see a `JOBID` use that with the following command to start a shell in the machine you requested

```bash
srun --jobid=JOBID --export=HOME,TERM,SHELL --pty /bin/bash --login
```

Once you gain access to the shell, go to the project directory and run the container 

```bash
cd /projects/academic/wenyaoxu/anarghya/research/cerebro/

apptainer exec --nv \
    /projects/academic/wenyaoxu/anarghya/container-directory/eeg2025.sif \
    python [your_script]
```

You can run the scripts using this method, but you will have to keep your terminal open so this method is suited for testing or debugging not keeping an experiment running
### Use this if you want to run a long experiment

You can update the parameters on top similar to the previous section, and also add your email if you want email notifications. 

```bash
#!/bin/bash -l

#SBATCH --job-name="cerebro-training"
#SBATCH --cluster=ub-hpc
#SBATCH --partition=arm64
#SBATCH --qos=arm64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gpus-per-node=gh200:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/ctc_log-%j.out
#SBATCH --mail-user=[your email]
#SBATCH --mail-type=FAIL

# Change to the working directory
cd /projects/academic/wenyaoxu/anarghya/research/cerebro/

# Run the script with srun
srun apptainer exec --nv \
    /projects/academic/wenyaoxu/anarghya/container-directory/eeg2025.sif \
    python [your_script_]
```

Save this as a file for example `training.script` then run the following command

```
sbatch training.script
```

### Using VSCode in CCR

This uses the VSCode tunnel executable that is in the shared directory. We will have to test if this works across users. If it doesn't, you will have to download the VSCode executable and use that in this script instead.

```bash
#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=arm64
#SBATCH --qos=arm64
#SBATCH --gpus-per-node=gh200:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/server.out
#SBATCH --mail-user=[email]
#SBATCH --mail-type=fail

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"

# Change to the working directory
cd /projects/academic/wenyaoxu/anarghya/ || exit 1

# Accept image file as first argument or environment variable
IMAGE_FILE=${1:-${SIF_IMAGE:-torch-arm64.sif}}

echo "Using container image: $IMAGE_FILE"

# Run the script with srun
apptainer exec --nv -B /scratch/:/scratch/ \
    "/projects/academic/wenyaoxu/anarghya/container-directory/${IMAGE_FILE}" \
    ./code-arm64 tunnel
```

To run this script (`run-server.script`), use the following 

```bash
sbatch run-server.script eeg2025.sif
```

On the first run, you will have to sign in with your account. To do that, see the logs file `cat logs/server`. There will be a link out there; follow it and sign in. Once that is complete, you should be able to see the connection in your VSCode under the remote tunnels section.

### General Commands 

To look at the currently running jobs in CCR

```bash
squeue -M all -u $LOGNAME
```

To cancel a running job

```bash
scancel <job_id>
```

