# Running Procedure

The same running procedure applies to all three cases: single-cylinder flow, flow past two staggered cylinders, and backward-facing step flow with cylindrical obstacles.

## Step 1. Configure the Simulation

Open the corresponding project folder in PyCharm and select the configured Python interpreter with CUDA support.

Open `server_settings.py` and configure the simulation parameters:

```python
ACTION = "prepare"
DEVICE = "cuda:0"
RE_LIST = [20., 40., 100., 150.]
REBUILD_MESH = False
RESUME_CHECKPOINT = None
INITIALIZE_CHECKPOINT = None
```

Set `EXPERIMENT` to the experiment name defined for the corresponding case. Modify `RE_LIST` according to the required Reynolds numbers.

## Step 2. Prepare the Simulation

Set:

```python
ACTION = "prepare"
```

Run `run_server.py` in PyCharm.

The program performs environment and CUDA checks, generates the mesh if necessary, verifies mesh quality and boundary conditions, and produces mesh visualization files. No training is performed at this stage.

## Step 3. Run the Simulation

After successful preparation, change:

```python
ACTION = "train"
```

Run `run_server.py` again.

The simulation starts from the initial time step and performs neural-operator-based time advancement, physical residual evaluation, local physics correction, and checkpoint saving.

## Step 4. Resume an Interrupted Simulation

If the simulation is interrupted, modify `server_settings.py`:

```python
ACTION = "resume"
RESUME_CHECKPOINT = r"path\to\checkpoint.pt"
```

Run `run_server.py` to continue from the saved checkpoint.

## Step 5. Access the Simulation Results

Simulation results are saved in the corresponding project's `runs` directory, including numerical logs, checkpoints, flow-field data, and generated figures.

The same procedure is followed independently for all three cases, using their respective project folders and simulation configurations.
