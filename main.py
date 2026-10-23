import argparse
import os
from pathlib import Path

from features_generation import feature_extraction_rev, label_extractor, generate_label_fasta, template_fasta_generator
from inference_code import base_inference_new


def main():
    parser = argparse.ArgumentParser(
        description="CARTnet: Contact-Aware B-cell Epitope Prediction"
    )

    parser.add_argument(
        "--input_struc",
        type=str,
        required=True,
        help="Path to input antigen structure (.pdb)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save prediction results"
    )


    parser.add_argument(
        "--template_struc",
        type=str,
        default=None,
        help="Optional template structure (.pdb)"
    )

    args = parser.parse_args()

    # Convert paths to absolute paths
    input_struc = Path(args.input_struc).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    # Validate input structure
    if not input_struc.is_file():
        parser.error(f"Input structure does not exist: {input_struc}")
    # Validate optional template structure
    template_struc = None
    if args.template_struc is not None:
        template_struc = Path(args.template_struc).expanduser().resolve()
        if not template_struc.is_file():
            parser.error(f"Template structure does not exist: {template_struc}")
    # Validate output directory
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"Output path exists but is not a directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_antigen_path = feature_extraction_rev.main(str(input_struc), str(output_dir) + "/")
    target = os.path.basename(input_struc).split(".")[0]
    # Select inference mode
    if template_struc is not None:
        print("Running CARTnet template-guided inference")
        # pass template stuct to label extractor
        template_dir = str(output_dir)+"/template/"
        extracted_label = label_extractor.main(template_struc,template_dir)
        #then pass the info to label fasta
        temp_initial_label_fasta_file = generate_label_fasta.main(template_dir,template_struc)
        #then genrate all feture

        template_fasta = template_fasta_generator.main(temp_initial_label_fasta_file,fasta_antigen_path,str(output_dir)+"/")
        #then get temp fasta
        #then pass to inference
        base_inference_new.main(
            str(output_dir) + "/",
            target,
            template_fasta
        )

    else:
        print("Running CARTnet basic inference")
        base_inference_new.main(str(output_dir)+"/",target,"")

    print(  str(input_struc),     str(output_dir),     str(template_struc))
if __name__ == "__main__":
    main()