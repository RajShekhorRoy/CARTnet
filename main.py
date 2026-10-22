import argparse
import os
from pathlib import Path

from features_generation import feature_extraction_rev
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

    # Select inference mode
    if template_struc is not None:
        print("Running CARTnet template-guided inference")

        # template_inference(
        #     str(input_struc),
        #     str(output_dir),
        #     str(template_struc)
        # )

    else:
        print("Running CARTnet basic inference")
        print(output_dir)
        feature_extraction_rev.main(str(input_struc),str(output_dir)+"/")
        target =os.path.basename(input_struc).split(".")[0]
        base_inference_new.main(str(output_dir)+"/",target,False)

    print(  str(input_struc),     str(output_dir),     str(template_struc))
if __name__ == "__main__":
    main()