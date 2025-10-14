#!/bin/bash

# Define the command to run
command="python batch.py"

# The newline characters are \r\n or ^M
# tr replaces both \r and \n; the squeeze option removes the extra |'s
messages=$(cat filter_messages.txt | tr -s '\r\n' '|')

# for debugging
# echo "$messages"

# Run the command, replace tabs with spaces in the output, and filter it in real-time
#$command | grep -Ev "$messages" | cat -e
$command | grep -Ev "$messages"