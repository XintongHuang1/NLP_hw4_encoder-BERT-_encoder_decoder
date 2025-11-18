"""

Script to evaluate a specific checkpoint that didn't finish training

"""

import os

import argparse

import torch

from tqdm import tqdm



from t5_utils import initialize_model, load_model_from_checkpoint

from transformers import T5TokenizerFast

from load_data import load_t5_data

from utils import compute_metrics, save_queries_and_records



DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def get_args():

    parser = argparse.ArgumentParser(description='Evaluate a saved checkpoint')

    parser.add_argument('--checkpoint_dir', type=str, required=True,

                        help='Path to checkpoint directory (e.g., checkpoints/ft_experiments/exp4_lr3e4)')

    parser.add_argument('--experiment_name', type=str, required=True,

                        help='Experiment name for output files (e.g., exp4_lr3e4)')

    parser.add_argument('--finetune', action='store_true', 

                        help='Whether this was a finetuned model')

    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--test_batch_size', type=int, default=16)

    return parser.parse_args()



def eval_epoch(model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path):

    model.eval()

    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    

    model_queries = []

    with torch.no_grad():

        for batch in tqdm(dev_loader, desc="Evaluating dev set"):

            encoder_input, encoder_mask, _, _, _ = batch

            encoder_input = encoder_input.to(DEVICE)

            encoder_mask = encoder_mask.to(DEVICE)



            gen_outputs = model.generate(

                input_ids=encoder_input,

                attention_mask=encoder_mask,

                max_new_tokens=300,

                num_beams=8,

                early_stopping=True,

                length_penalty=0.95,

                decoder_start_token_id=0

            )



            for seq in gen_outputs:

                query = tokenizer.decode(seq, skip_special_tokens=True)

                model_queries.append(query.strip())



    save_queries_and_records(model_queries, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(

        gt_sql_path, model_sql_path, gt_record_path, model_record_path

    )

    

    error_rate = sum(1 for m in model_error_msgs if m != "") / len(model_error_msgs) if model_error_msgs else 0.0

    return record_f1, record_em, sql_em, error_rate



def test_inference(model, test_loader, model_sql_path, model_record_path):

    model.eval()

    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')



    model_queries = []

    with torch.no_grad():

        for batch in tqdm(test_loader, desc="Running test inference"):

            if len(batch) == 3:

                encoder_input, encoder_mask, _ = batch

            else:

                encoder_input, encoder_mask, _, _, _ = batch



            encoder_input = encoder_input.to(DEVICE)

            encoder_mask = encoder_mask.to(DEVICE)



            gen_outputs = model.generate(

                input_ids=encoder_input,

                attention_mask=encoder_mask,

                max_new_tokens=300,

                num_beams=8,

                early_stopping=True,

                length_penalty=0.95,

                decoder_start_token_id=0

            )



            for seq in gen_outputs:

                query = tokenizer.decode(seq, skip_special_tokens=True)

                model_queries.append(query.strip())



    save_queries_and_records(model_queries, model_sql_path, model_record_path)

    print(f"Saved {len(model_queries)} test queries to {model_sql_path}")



def main():

    args = get_args()

    

    # Create results/records directories if needed

    os.makedirs('results', exist_ok=True)

    os.makedirs('records', exist_ok=True)

    

    # Load data

    print("Loading data...")

    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)

    

    # Load model from checkpoint

    print(f"Loading model from {args.checkpoint_dir}...")

    args.checkpoint_dir = args.checkpoint_dir  # Set for load_model_from_checkpoint

    model = load_model_from_checkpoint(args, best=True)

    model.eval()

    model.to(DEVICE)

    

    model_type = 'ft' if args.finetune else 'scr'

    

    # Evaluate on dev set

    print("\nEvaluating on dev set...")

    gt_sql_path = 'data/dev.sql'

    gt_record_path = 'records/ground_truth_dev.pkl'

    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_dev.sql'

    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_dev.pkl'

    

    dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(

        model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path

    )

    

    print(f"\n{'='*60}")

    print(f"Dev set results:")

    print(f"  Record F1:  {dev_record_f1:.4f}")

    print(f"  Record EM:  {dev_record_em:.4f}")

    print(f"  SQL EM:     {dev_sql_em:.4f}")

    print(f"  Error Rate: {dev_error_rate*100:.2f}%")

    print(f"{'='*60}")

    

    # Run inference on test set

    print("\nRunning inference on test set...")

    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_test.sql'

    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_test.pkl'

    test_inference(model, test_loader, model_sql_path, model_record_path)

    

    # Check if this is the best model

    best_f1_file = 'best_f1.txt'

    previous_best_f1 = 0.0

    

    if os.path.exists(best_f1_file):

        try:

            with open(best_f1_file, 'r') as f:

                previous_best_f1 = float(f.read().strip())

            print(f"\nPrevious best F1: {previous_best_f1:.4f}")

        except:

            previous_best_f1 = 0.0

    

    print(f"Current F1:       {dev_record_f1:.4f}")

    

    if dev_record_f1 > previous_best_f1:

        print(f"\n🎉 NEW BEST MODEL! F1 improved from {previous_best_f1:.4f} to {dev_record_f1:.4f}")

        

        # Update best F1 record

        with open(best_f1_file, 'w') as f:

            f.write(f"{dev_record_f1:.6f}")

        

        # Copy test results to submission filenames

        submission_sql = f'results/t5_{model_type}_test.sql'

        submission_pkl = f'records/t5_{model_type}_test.pkl'

        

        import shutil

        shutil.copy(model_sql_path, submission_sql)

        shutil.copy(model_record_path, submission_pkl)

        print(f"✓ Updated submission files: {submission_sql}, {submission_pkl}")

        

        # Save best model info

        with open('best_model_info.txt', 'w') as f:

            f.write(f"Best Model Info\n")

            f.write(f"=" * 50 + "\n")

            f.write(f"Experiment: {args.experiment_name}\n")

            f.write(f"Checkpoint: {args.checkpoint_dir}\n")

            f.write(f"Model Type: {'finetuned' if args.finetune else 'from scratch'}\n")

            f.write(f"Dev Record F1: {dev_record_f1:.4f}\n")

            f.write(f"Dev Record EM: {dev_record_em:.4f}\n")

            f.write(f"Dev SQL EM: {dev_sql_em:.4f}\n")

            f.write(f"Error Rate: {dev_error_rate*100:.2f}%\n")

        print(f"✓ Saved model info to: best_model_info.txt")

    else:

        print(f"\n❌ No improvement. Previous best: {previous_best_f1:.4f}, Current: {dev_record_f1:.4f}")



if __name__ == "__main__":

    main()


