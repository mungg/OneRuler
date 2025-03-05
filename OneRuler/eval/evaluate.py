import argparse
import os
import json
import re
import unicodedata

none_dict = {
    "en": ["none"],
    "ko": ["없음"],
    "pl": ["brak"],
    "zh": ["无"],
    "vi": ["Không có"],
    "ja": ["なし","数字はありません"],
    "ta": ["ஏதுமில்லை"],
    "hu": ["nincs"],
    "fr": ["aucun"],
    "no": ["ingen"],
    "uk": ["немає", "Нема"],
    "ru": ["нет"],
    "de": ["Keine vorhanden"],
    "es": ["ninguno"],
    "sv": ["inga"],
    "fi": ["ei mikään"],
    "cs": ["žádné", "žádná"],
    "sr": ["nema"],
    "pt": ["nenhum"],
    "it": ["nessuno"],
    "fa": ["هیچ کدام"],
    "sw": ["hakuna"],
    "nl": ["geen"],
    "st": ["ha ho letho"],
    "hi": ["कोई नहीं"],
    "da": ["ingen"]
}



def extract_model_name(file_path):
    """Extract model_name from the given file path."""
    file_name = os.path.basename(file_path)
    if 'prediction_new' in file_name:
        model_name = file_name.split("prediction_new_")[-1].split(".jsonl")[0]
    else:
        model_name = file_name.split("prediction_")[-1].split(".jsonl")[0]
    return model_name

def clean_text(text):
    return text.strip().lower().replace('\u200c', '').replace(' ', '')

def escape_quotes(text):
    return text.replace('"', '""')

def is_correct_order(correct_answer, model_answer):
    model_answer = model_answer.strip().lower()
    correct_answer = [correct.strip().lower() for correct in correct_answer]
    return all(word in model_answer for word in correct_answer)

def compare_numbers(lang, correct_answer, model_answer):    
    if '-' in lang:
        inst_lang = lang.split("-")[1]
    else:
        inst_lang = lang
    if not model_answer:
        return False
    processed_model_answer = unicodedata.normalize("NFKC", model_answer)
   
    none_words = none_dict[inst_lang]
    # Step 1.5: Check if any word in none_words is present in the processed answer; if yes, auto-fail.
    for word in none_words:
        if word in processed_model_answer or clean_text(word) in processed_model_answer:
            return False
    
    # Step 2: Extract all numeric substrings from the processed answer.
    numeric_strings = re.findall(r'\d+', processed_model_answer)

    # Step 3: Remove numbers that consist of a single digit.
    numeric_strings = [num for num in numeric_strings if len(num) > 1]

    # Step 4: Remove duplicates while preserving the original order.
    numeric_strings = list(dict.fromkeys(numeric_strings))

    # If no numerics are found after processing, return "0".
    if not numeric_strings:
        return False

    # Step 5: Convert the extracted number strings to integers.
    try:
        extracted_numbers = [int(num) for num in numeric_strings]
    except Exception:
        return False

    # Convert correct_answers elements to integers to ensure numeric comparison.
    try:
        correct_converted = [int(item) for item in correct_answer]
    except Exception:
        return False

    # Step 6: Check that the number of extracted numbers matches the length of correct_answers.
    if len(extracted_numbers) != len(correct_converted):
        return False

    # Step 7: Compare the extracted numbers with the correct answers element-wise.
    if set(extracted_numbers) == set(correct_converted):
        return True
    else:
        return False

def compare_none(lang, correct_answer, model_answer):   
    # Lower-case all inputs for consistent, case-insensitive processing.
    if '-' in lang:
        inst_lang = lang.split("-")[1]
    else:
        inst_lang = lang

    processed_model_answer = clean_text(unicodedata.normalize("NFKC", model_answer))
    correct_answer = [clean_text(answer) for answer in correct_answer]
    none_words = [clean_text(word) for word in none_dict[inst_lang]]

    # Step 2: Remove single digit numbers from the processed answer.
    processed_model_answer = re.sub(r'\b\d\b', '', processed_model_answer)

    # Step 3: Extract all numeric sequences from the processed answer.
    numbers = re.findall(r'\d+', processed_model_answer)

    # If any multi-digit number is found, return "0".
    # if numbers:
    #     return False
    # Step 4: Check if any of the words in none_words are present.
    for word in none_words:
        if word in processed_model_answer:
            return True

    # If none of the none_words are found, return "0".
    return False


def evaluate_jsonl(file_path, task, lang, model_name):
    """Evaluate JSONL file by comparing outputs and response-{model_name}."""
    accs = []
    set_index = set(list(range(1, 50)))
    index_list = []
    count = 0
    error_cases= []
    reasoning_tokens_true = []
    reasoning_tokens_false = []
    with open(file_path, 'r', encoding='utf-8', errors="replace") as f:
        for line in f:
            data = json.loads(line)
            response_keys = [key for key in data.keys() if key.startswith("response-")]
            if len(response_keys) != 1:
                print(f'problem in {response_keys}')
                exit()
            if data['index'] in index_list:
                print("duplicated - SKIP")
                continue
            if 'o3-' in model_name:
                if data['finish_reason'] == 'length':
                    continue
            response_key = response_keys[0]
            if not data[response_key]:
                if data[response_key] == None:
                    print("there is no response")
                    continue
                
            index_list.append(data['index'])
            if 'outputs' in data and response_key in data:
                reference = data['outputs']
                response = data[response_key]

                ## preprocessing
                if "deepseek" in model_name:
                    match = re.search(r"<Answer>.*?</Answer>", response)
                    if match:
                        extracted_value = match.group(0)
                        hypothesis = extracted_value 
                    else:
                        hypothesis = response
                else:
                    hypothesis = response
                if 'niah_none' in task:
                    acc = compare_none(lang, reference, hypothesis)
                elif 'cwe' in task:
                    acc = is_correct_order(reference, hypothesis)
                else:
                    acc = compare_numbers(lang, reference, hypothesis)
                if not acc:
                    hypothesis_clean = hypothesis.replace("\n", "")
                    error_cases.append([task, model_name, reference, hypothesis_clean, response])
                accs.append(acc)
                reasoning = data.get('reasoning_tokens', None) 
                if data.get('reasoning_tokens', None):
                    if acc:
                        reasoning_tokens_true.append(reasoning)
                    else:
                        reasoning_tokens_false.append(reasoning)
            else:
                print(f"wrong form")
                continue
            count += 1
    no_prediction_index = set_index - set(index_list)
    print(f"Total processing data: {count}, {len(no_prediction_index)} are omitted prediction. indices: {no_prediction_index}")    
    avg_acc = sum(accs) / len(accs) if accs else 0
    return {
        "model_name": model_name,
        "avg_acc": avg_acc,
        "processed": count,
        'reasoning_tokens_true': sum(reasoning_tokens_true) / len(reasoning_tokens_true) if reasoning_tokens_true else None,
        'reasoning_tokens_false': sum(reasoning_tokens_false) / len(reasoning_tokens_false) if reasoning_tokens_false else None
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs and save results")
    parser.add_argument("--input_path", required=True, help="Path to the input file to evaluate")
    parser.add_argument("--task", help="Task being evaluated")
    parser.add_argument("--language", help="Language of the evaluation")
    parser.add_argument("--model_name", help="Name of the model being evaluated")

    args = parser.parse_args()
    result = evaluate_jsonl(args.input_path, args.task, args.language, args.model_name)
    result.update({"language": args.language, "task":  args.task})
    print(result)



if __name__ == "__main__":
    main()