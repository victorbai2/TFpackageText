#!/bin/bash

Usage()
{
	# Display Usage
	echo
	echo "Syntax: options: [h|m|f|i|j]"
	echo "Options                 Definition"
	echo "-h --help               print the help."
	echo "-m --mode               specify train/eval/pred."
	echo "-f --filename           provide the filename."
	echo "-i --input_shape        the input dimention shape."
	echo
}

while getopts ":hm:f:i:" option; do
	case $option in
		h) # display Usage
			Usage
			exit;;
		m) #run mode train/eval/pred.
			mode=$OPTARG;;
		f) #filename
			file=$OPTARG;;
		i) #input_shape
			input_shape=$OPTARG;;
		\?) # incorrect option
			Usage
			echo "Error: Invalid option"
			exit;;
	esac
done


if [[ !( $mode == "train" || $mode == "eval" || $mode == "pred" ) ]]
then
	echo "No mode is given, please input first argument "train" or "eval" or "pred". "
	exit 0
elif [ -z $file ]
then
	echo "No file is given, please input second argument "file" name. "
	exit 0
#elif [[ ($mode == "pred")]]
#then
#	if [[ (-z $input_shape )]]
#	then
#		echo  "please provide the two arguments for input_shape, eg. startup.sh -mode pred -f filename -i '1 784'"
#	else
#		echo "run $mode on $file with input_shape: $input_shape"
#			python $file \
#				--mode $mode --input_shape $input_shape
#	fi
else
	echo "run $mode on $file"
		python $file \
			--mode $mode
fi

echo
