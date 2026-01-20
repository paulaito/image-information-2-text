from visual_info_extractor.io import DataIO
from visual_info_extractor.inferencer import VLMInferencer
from visual_info_extractor.logger import logging
from visual_info_extractor.config import EvaluatorConfig
from visual_info_extractor.ollama.healthcheck import OllamaHealthChecker
from visual_info_extractor.ollama.client import OllamaClient

import pandas as pd
import json
import glob

class Evaluator:
    def __init__(self, config_path: str):
        self.io = DataIO()
        self.config = EvaluatorConfig.from_yaml(config_path)
        self.version = self.config.version

        self.path_to_eval = self.config.input_dir + "/" + self.version
        self.output_path = self.config.output_dir + "/" + self.version

        self.set_client()
        self.evaluator_model = self.config.evaluator_model

        self.client.pull_models([self.evaluator_model])


    def set_client(self):
        connection_status = OllamaHealthChecker(host=self.config.ollama_host).check_connection()
        if connection_status:
            self.client = OllamaClient(host=self.config.ollama_host)

    def read_data(self):
        self.df_to_eval = self.io.read_directory(
            self.path_to_eval,
        )

    @staticmethod
    def build_judge_prompt(prompt, groundtruth, response):
        return f"""
        You are a strict but fair multimodal evaluator.

        Your task is to judge whether an assistant response is a correct answer to
        the USER PROMPT, given the IMAGE and the GROUND TRUTH.

        ROLES:
        - The image is the primary source of observable facts.
        - The ground truth provides minimum guaranteed facts and must not be contradicted.
        - The prompt defines the expected scope, level of detail, and whether interpretation is allowed.

        RULES:
        1. The response must correctly answer the prompt.
        2. The response must be consistent with the ground truth.
        3. The response may include additional details beyond the ground truth ONLY if they are clearly visible in the image.
        4. Do NOT allow hallucinations, speculation, inferred intent, or unsupported interpretation.
        5. If you suspect an hallucination, validate it against the image.
        5. Grammatical or spelling differences (e.g., "color" vs "colour", capitalization) do NOT affect correctness.
        6. If you are uncertain whether a detail is supported, mark FALSE.

        ---

        PROMPT GIVEN TO THE ASSISTANT:
        \"\"\"{prompt}\"\"\"

        GROUND TRUTH:
        \"\"\"{groundtruth}\"\"\"
            
        IMAGE: received along with this prompt.

        ASSISTANT RESPONSE:
        \"\"\"{response}\"\"\"

        Return ONLY a JSON object with:
        - label: TRUE or FALSE
        - reason: one short sentence explaining the decision
        """

    @staticmethod
    def parse_judge_output(content: str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "label": "INVALID_JSON",
                "reason": "INVALID_JSON"
            }

    def evaluate(self) -> None:
        num_runs = len(self.df_to_eval)
        eval_results = []
        value_errors = 0
        run = 0
        logging.info("Starting evaluation...")
        logging.info(f"LLM Evaluator Model: {self.evaluator_model}")

        for idx, row in self.df_to_eval.iterrows():
            run += 1
            if run > 2:
                break
            logging.info(f"Evaluating run: {run}/{num_runs}")
            logging.info(f"Model: {row['model']}")
            logging.info(f"Image path: {row['image_path']}")
            judge_prompt = self.build_judge_prompt(
                prompt=row["prompt"],
                groundtruth=row["groundtruth"],
                response=row["response"],
            )

            try:
                content, _ = self.client.run_chat(
                    model=self.evaluator_model,
                    prompt=judge_prompt,
                    image_paths=["../" + row["image_path"]],
                )

                judge = self.parse_judge_output(content)

                eval_results.append({
                    "model": row["model"],
                    "prompt": row["prompt"],
                    "groundtruth": row["groundtruth"],
                    "response": row["response"],
                    "image_path": row["image_path"],
                    "judge_label": judge["label"],
                    "judge_reason": judge["reason"],
                })

            except ValueError as e:
                logging.error(f"ValueError for row {idx}: {e}")
                value_errors += 1
                continue

        logging.info(f"Total ValueErrors encountered: {value_errors}")
        
        return pd.DataFrame(eval_results), value_errors

    def export_results(self, results_df: pd.DataFrame) -> None:
        self.io.write(
            df = results_df,
            file_name=f"evaluation_results.csv",
            output_dir=self.output_path,
            append=False
        )

    def run(self) -> None:
        self.read_data()
        results_df, value_errors = self.evaluate()
        self.export_results(results_df)