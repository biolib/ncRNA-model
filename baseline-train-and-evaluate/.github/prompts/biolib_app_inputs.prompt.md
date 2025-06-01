# Main task
Your task is to make sure that all inputs are handled correctly in the give Python script and/or biolib config file.

Inputs are defined in the [config.yml](../../.biolib/config.yml) file and the main Python script, usually found in [run.py](../../run.py) or [main.py](../../main.py).

# Syntax of config.yml
The file config.yml contains the information needed to render and run an application on BioLib. This configuration defines the entry to your application and what input arguments the user can set. When you edit an application using the graphical interface on BioLib the config.yml file is automatically updated.

The config file and python script specify how input options and settings will be rendered to the user of the application, and how inputs will be parsed. The input field should follow this structure:

```
arguments:
    -   key: --data # required
        description: 'Input Dropdown' # required
        key_value_separator: ' ' # optional, default is ' '
        default_value: '' # optional, default is ''
        type: dropdown # required
        options:
            'This will be shown as option one': 'value1'
            'This will be shown as option two': 'value2'
        required: true # optional, default is true
```

Under `type` you have the following options:

* `text` provides a text input field
* `file` provides a file select where users can upload an input file
* `text-file` provides both a text input field and a file select allowing the user supply either
* `sequence` like text-file, with checks for valid FASTA input characters
* `hidden` allows the application creator to provide a default input argument without it being shown to the end-user
* `toggle` provides a toggle switch where users can choose two options. Note that the options need to be named 'on' : 'value1' and 'off': 'value2'
* `number` provides a number input field
* `radio` provides a "radio select" where users can select one amongst a number of prespecified options
* `dropdown` provides a dropdown menu where users can select one amongst a number of prespecified options
* `multiselect` provides a dropdown menu where users can select one or more prespecified options

`sub_arguments`: Allow you to specify arguments that are only rendered if a user chooses a particular option in the parent argument. For example, an application might allow the user to run one of two commands, where each of these commands would need different input arguments:

```
arguments:
    -   key: --function
        description: 'Choose a function'
        key_value_separator: ''
        default_value: ''
        type: dropdown
        options:
            'Command A': a
            'Command B': b
        sub_arguments:
            a:
                -   key: --argument_a
                    description: "Argument A takes a file input"
                    type: file
            b:
                -   key: --argument_b
                    description: 'Argument B takes a text input'
                    type: text
```

Inputs in the Python script should be parsed with argparse, and should also enshrine the same requirements and defaults such that use is identical between the frontend and Python.