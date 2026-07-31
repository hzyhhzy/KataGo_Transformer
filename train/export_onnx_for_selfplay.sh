#!/bin/bash -eu
set -o pipefail
{
#Takes any models in torchmodels_toexport/ and outputs a cuda-runnable model file to modelstobetested/
#Takes any models in torchmodels_toexport_extra/ and outputs a cuda-runnable model file to models_extra/
#Should be run periodically.

function printUsage() {
    echo "Usage: $0 NAMEPREFIX BASEDIR USEGATING -pos-len N [-simplify] [-non-swa]"
    echo "Currently expects to be run from within the 'train' directory of the KataGo repo, or otherwise in the same dir as export_onnx.py."
    echo "NAMEPREFIX string prefix for this training run, try to pick something globally unique. Will be displayed to users when KataGo loads the model."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "USEGATING = 1 to use gatekeeper, 0 to not use gatekeeper and output directly to models/"
    echo "-pos-len N is required and sets the exported board edge length"
    echo "-simplify keeps only the simplified ONNX model and fails if simplification does not produce one"
    echo "-non-swa exports the ordinary model instead of the SWA model"
}

if [[ $# -lt 3 ]]
then
    printUsage >&2
    exit 1
fi
NAMEPREFIX="$1"
shift
BASEDIR="$1"
shift
USEGATING="$1"
shift

POS_LEN=""
SIMPLIFY=0
USE_SWA=1

while [[ $# -gt 0 ]]
do
    case "$1" in
        -pos-len)
            if [[ $# -lt 2 ]]
            then
                echo "Error: -pos-len requires a value" >&2
                printUsage >&2
                exit 1
            fi
            POS_LEN="$2"
            shift 2
            ;;
        -simplify)
            SIMPLIFY=1
            shift
            ;;
        -non-swa)
            USE_SWA=0
            shift
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            printUsage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$POS_LEN" ]]
then
    echo "Error: -pos-len is required" >&2
    printUsage >&2
    exit 1
fi
if [[ ! "$POS_LEN" =~ ^[1-9][0-9]*$ ]]
then
    echo "Error: -pos-len must be a positive integer, got: $POS_LEN" >&2
    exit 1
fi

#------------------------------------------------------------------------------

mkdir -p "$BASEDIR"/torchmodels_toexport
mkdir -p "$BASEDIR"/torchmodels_toexport_extra
mkdir -p "$BASEDIR"/modelstobetested
mkdir -p "$BASEDIR"/models_extra
mkdir -p "$BASEDIR"/models

function exportStuff() {
    FROMDIR="$1"
    TODIR="$2"

    #Sort by timestamp so that we process in order of oldest to newest if there are multiple
    for FILEPATH in $(find "$BASEDIR"/"$FROMDIR"/ -mindepth 1 -maxdepth 1 -printf "%T@ %p\n" | sort -n | cut -d ' ' -f 2)
    do
        #Make sure to skip tmp directories that are transiently there by the training,
        #they are probably in the process of being written
        if [ ${FILEPATH: -4} == ".tmp" ]
        then
            echo "Skipping tmp file:" "$FILEPATH"
        elif [ ${FILEPATH: -9} == ".exported" ]
        then
            echo "Skipping self tmp file:" "$FILEPATH"
        elif [ ${FILEPATH: -18} == ".ipynb_checkpoints" ]
        then
            echo "Skipping jupyter lab dir:" "$FILEPATH"
        else
            echo "Found model to export:" "$FILEPATH"
            NAME="$(basename "$FILEPATH")"

            SRC="$BASEDIR"/"$FROMDIR"/"$NAME"
            TMPDST="$BASEDIR"/"$FROMDIR"/"$NAME".exported
            TARGET="$BASEDIR"/"$TODIR"/"$NAME"

            if [ -d "$BASEDIR"/modelstobetested/"$NAME" ] ||  \
               [ -d "$BASEDIR"/rejectedmodels/"$NAME" ] || \
               [ -d "$BASEDIR"/models/"$NAME" ] || \
               [ -d "$BASEDIR"/models_extra/"$NAME" ] || \
               [ -d "$BASEDIR"/modelsuploaded/"$NAME" ]
            then
                echo "Model with same name aleady exists, so skipping:" "$SRC"
            else
                rm -rf "$TMPDST"
                mkdir "$TMPDST"

                EXPORT_ARGS=(
                    -checkpoint "$SRC"/model.ckpt
                    -export-dir "$TMPDST"
                    -model-name "$NAMEPREFIX""-""$NAME"
                    -filename-prefix model
                    -skip-verification
                    -pos-len "$POS_LEN"
                )
                if [[ "$USE_SWA" -eq 1 ]]
                then
                    EXPORT_ARGS+=(-use-swa)
                fi
                if [[ "$SIMPLIFY" -eq 1 ]]
                then
                    EXPORT_ARGS+=(-simplify)
                fi

                set -x
                set +e
                python ./export_onnx.py "${EXPORT_ARGS[@]}"
                EXPORT_STATUS=$?
                set -e

                if [[ "$SIMPLIFY" -eq 1 ]]
                then
                    # Never leave the ordinary model behind when simplification
                    # was requested, even when export_onnx.py fails partway through.
                    rm -f "$TMPDST"/model.onnx
                    if [[ "$EXPORT_STATUS" -ne 0 ]]
                    then
                        set +x
                        echo "Error: ONNX export or simplification failed with status $EXPORT_STATUS; ordinary model.onnx was deleted" >&2
                        return "$EXPORT_STATUS"
                    fi
                    if [[ ! -f "$TMPDST"/model_simplified.onnx ]]
                    then
                        set +x
                        echo "Error: simplification did not produce model_simplified.onnx; ordinary model.onnx was deleted" >&2
                        return 1
                    fi
                    mv "$TMPDST"/model_simplified.onnx "$TMPDST"/model.onnx
                elif [[ "$EXPORT_STATUS" -ne 0 ]]
                then
                    set +x
                    echo "Error: ONNX export failed with status $EXPORT_STATUS" >&2
                    return "$EXPORT_STATUS"
                fi

                python ./clean_checkpoint.py \
                        -checkpoint "$SRC"/model.ckpt \
                        -output "$TMPDST"/model.ckpt
                set +x

                rm -r "$SRC"

                #Make a bunch of the directories that selfplay will need so that there isn't a race on the selfplay
                #machines to concurrently make it, since sometimes concurrent making of the same directory can corrupt
                #a filesystem
                #Only when not gating. When gating, gatekeeper is responsible.
                if [ "$USEGATING" -eq 0 ]
                then
                    if [ "$TODIR" != "models_extra" ]
                    then
                        mkdir -p "$BASEDIR"/selfplay/"$NAME"
                        mkdir -p "$BASEDIR"/selfplay/"$NAME"/sgfs
                        mkdir -p "$BASEDIR"/selfplay/"$NAME"/tdata
                        rm -f "$BASEDIR"/latest.onnx
                        cp -f "$TMPDST"/model.onnx "$BASEDIR"/latest.onnx
                    fi
                fi

                #Sleep a little to allow some tolerance on the filesystem
                sleep 5

                mv "$TMPDST" "$TARGET"
                echo "Done exporting:" "$NAME" "to" "$TARGET"
            fi
        fi
    done
}

if [ "$USEGATING" -eq 0 ]
then
    exportStuff "torchmodels_toexport" "models"
else
    exportStuff "torchmodels_toexport" "modelstobetested"
fi
exportStuff "torchmodels_toexport_extra" "models_extra"

exit 0
}
