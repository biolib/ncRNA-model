# BioLib Run Apps Prompt
Your task is to run some kind of apps at the users discretion. Here are instructions on how running an app works.

## Login

You need to be logged in, unless this is running as part of a different app. To log in with your BioLib account in a Python notebook run the code below and follow the instructions shown:

```python
import biolib
biolib.login()
```

Alternatively, you can use an API token and set it as the `BIOLIB_TOKEN` environment variable. If the user requests this, direct them [here](https://biolib.com/settings/api-tokens/).

## Run using .cli()

To load an application into your Python script, add the following:

```python
import biolib
app = biolib.load('author/application')
```

To run an application call the function `.cli()` on the application you loaded above. For instance, to run samtools with the `--help` command:

```python
import biolib

samtools = biolib.load('samtools/samtools')
job = samtools.cli(args='--help')
print(job.get_stdout().decode())
```

Running an application returns a job object, which allows you to monitor progress and save results.

## Non blocking

By default, calling the function `.cli()` blocks until the application is finished. You can pass the keyword argument `blocking=False` to return immediately. For example the code below will print "in_progress":

```python
import biolib

samtools = biolib.load('samtools/samtools')
job = samtools.cli(args='--help', blocking=False)
print(job.get_status())
```

## Result prefix

You can annotate the result with a custom name when calling `.cli()` using the keyword argument `result_prefix` as:

```python
import biolib

samtools = biolib.load('samtools/samtools')
job = samtools.cli(args='--help', result_prefix='my_help_test')
```

Setting the result prefix makes it easy to distinguish results from one another on the result page.

## Run using .run()

The `.run()` function is a more Pythonic way to run applications where all keyword arguments are passed to the application as command line arguments. This function blocks and waits until the application is finished.

```python
samtools = biolib.load('samtools/samtools')
job = samtools.run()
```

## Run using .start()

The `.start()` function is a more Pythonic way to run applications where all keyword arguments are passed to the application as command line arguments. This function returns immediately when the job is created.

```python
samtools = biolib.load('samtools/samtools')
job = samtools.start()
```

## Search

To search for applications on BioLib use the function `biolib.search()` which takes a search query as the first argument:

```python
app_list = biolib.search('samtools')
print(app_list)
```

Should print something like below:

```
['samtools/samtools',
 'samtools/samtools-fixmate',
 'samtools/samtools-stats',
 'samtools/samtools-collate',
 'samtools/samtools-fastq',
 ...]
```

To run a specific application you can pass a value from the list above to `biolib.load()` and then call `app.cli()`:

```python
app = biolib.load(app_list[0])
job = app.cli('--help')
```

## Results

When a job has completed, its outputs can be accessed by the following functions:

```python
job.wait()          # Wait until done
job.get_stdout()    # Returns stdout as bytes
job.get_stderr()    # Returns stderr as bytes
job.get_exit_code() # Returns exit code of the application as an integer
```

## Save files to disk

To save the output files to a local directory like "result_files" run:

```python
job.save_files(output_dir='result_files')
```

The `.save_files()` function also takes an optional `path_filter` argument as a glob pattern. For example to save all `.pdb` files from a result you can run:

```python
job.save_files(output_dir='result_files', path_filter='*.pdb')
```

## In memory files

Work with result files without saving them to disk. To list the output files from a job:

```python
job.list_output_files()
```

To load a single file into memory, without saving it to disk, run:

```python
my_csv_file = job.get_output_file('/my_file.csv')
```

To pass an output file to a library like Pandas or BioPython, run `.get_file_handle()` on the object:

```python
import pandas as pd
my_dataframe = pd.read_csv(my_csv_file.get_file_handle())
```

## Jobs

A job object refers to a specific run of an application. It holds progress information of the application run and the result when the job has completed.

### List jobs

When signed in, you can print a table of your jobs by running:

```python
biolib.show_jobs(count=25)
```

where count refers to the number of jobs you want to show.

### Retrieve a job

To retrieve a Job in python call `biolib.get_job()` with the Job's ID.

```python
job = biolib.get_job(job_id)
print(job.get_status())
```

You can use this to determine if a job has completed or is still in progress.

### Open in browser

You can open the job in your web browser to view the graphical and interactive output files.

```python
job.open_browser()
```

### Stream output

If your Job is still running you can attach to its stdout and stderr by running:

```python
job.stream_logs()
```

This will print current output and keep streaming stdout and stderr until the job has finished.

### Download output files

You can download job output files using the job ID. The job ID can be found under "Details" on the Results page, or in the share link:

```python
job_id = '1a234567-b89...'
job = biolib.get_job(job_id)
job.save_files('job_output/')
```

### Download input files

To download the input files of a job:

```python
job_id = '1a234567-b89...'
job = biolib.get_job(job_id)
job.save_input_files(output_dir='input_files')
```

## Start jobs in parallel

Use the `blocking=False` argument to `.cli()` on an application to get the job immediately without having to wait for the application to finish.

This feature allows for parallelized workflows as the one below:

```python
samtools = biolib.load('samtools/samtools')
my_fasta_files = ['seq1.fasta', 'seq2.fasta']

my_jobs = []
for file in my_fasta_files:
    job = samtools.cli(file, blocking=False)
    my_jobs.append(job)
```

## Experiments

An Experiment is a collection of jobs that you can retrieve together. To group the jobs in an Experiment use the following syntax:

```python
with biolib.Experiment('my-experiment-name'):
    my_application.cli(input_1) # these two jobs will be
    my_application.cli(input_2) # grouped in the same Experiment
```

All jobs started under the with statement will be grouped under the Experiment's ID (in this case `my-experiment-name`).

### List experiments

When logged in, you can print a table of your experiments by running:

```python
biolib.show_experiments(count=10)
```

where count refers to the number of experiments you want to show.

### Retrieve an experiment

To load an Experiment in Python, run the following:

```python
my_experiment = biolib.get_experiment('my-experiment-name')
print(my_experiment)
```

### Wait for all jobs

To block and wait until all jobs of an experiment have finished, use the `.wait()` function:

```python
my_experiment.wait()
```

### Retrieve jobs

To get a list of the Job objects contained in an Experiment, run:

```python
my_jobs = my_experiment.get_jobs()
for job in my_jobs:
    # Print output
    if job.get_status() == 'completed':
        print(job.get_stdout())
    else:
        job.stream_logs()

    # Save output files
    job.save_files('my_results')
```

### List jobs in an Experiment

To show an overview of the jobs in your experiment run:

```python
my_experiment.show_jobs()
```

This prints a table of the jobs contained in your experiment.

### Mount files

Using `.mount_files()` you can mount all jobs of the experiment and their output files to a local directory. This allows you to explore all the files in the experiment using your file browser.

```python
my_experiment.mount_files(mount_path='my_local_directory')
