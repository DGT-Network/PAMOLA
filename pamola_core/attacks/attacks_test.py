"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Attack Simulation
-----------------------
This module provides an abstract base class for attack simulation feature
in PAMOLA.CORE. It defines the general structure and required methods for
implementing specific attack simulation

NOTE: This module requires 'numpy' and 'pandas' as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import os
import numpy as np
import pandas as pd
from pamola_core.attacks.attack_metrics import AttackMetrics
from pamola_core.attacks.attribute_inference import AttributeInference
from pamola_core.attacks.membership_inference import MembershipInference
from pamola_core.attacks.linkage_attack import LinkageAttack


class AttacksTest:
    """
    AttacksTest class for attack simulation in PAMOLA.CORE.
    This class define methods to test functions for attack simulation.
    """

    def __init__(
        self, attribute_inference, membership_inference, linkage_attack, attack_metrics
    ):
        self.attribute_inference = attribute_inference
        self.membership_inference = membership_inference
        self.linkage_attack = linkage_attack
        self.attack_metrics = attack_metrics

    def generate_name_abbreviations(self, full_names):
        """
        Create a list of name abbreviations from an initial list
        """

        abbreviations = []

        for name in full_names:
            words = name.split()  # Split the name into a list of words
            if len(words) > 1:
                abbreviation = (
                    " ".join([w[0] + "." for w in words[:-1]]) + " " + words[-1]
                )  # Abbreviate the first words, keep the last word the same
            else:
                abbreviation = words[0]  # If there is only one word, keep it as is

            abbreviations.append(abbreviation)

        return abbreviations

    def generate_city_abbreviations(self, names):
        """
        Create a list of city abbreviations from an initial list
        """

        abbreviations = []

        for name in names:
            words = name.replace(
                ",", ""
            ).split()  # Separate words, remove commas if any
            if len(words) > 1:
                abbreviation = "".join(
                    word[0] for word in words
                ).upper()  # Choose the first letter of each word
            else:
                abbreviation = name[
                    :3
                ].upper()  # If there is only one word, take the first three letters
            abbreviations.append(abbreviation)

        return abbreviations

    def read_data_from_csv(self, file_path):
        # file_path = r"C:\datatest\RandomPeople-3-clean 3.csv"
        # file_path = "data_files/RandomPeople.csv"
        return pd.read_csv(file_path)

    def generate_data(self, file_path=None):
        """
        Generate data for testing attack functions
        """

        if file_path is None:
            np.random.seed(42)
            # List of first_names
            first_names = [
                "John",
                "Alice",
                "David",
                "Emma",
                "Michael",
                "Sophia",
                "James",
                "Olivia",
                "Daniel",
                "Emily",
                "Robert",
                "Isabella",
                "William",
                "Mia",
                "Joseph",
                "Charlotte",
                "Charles",
                "Amelia",
                "Thomas",
                "Harper",
                "Henry",
                "Evelyn",
                "Christopher",
                "Abigail",
                "Matthew",
                "Ella",
                "Joshua",
                "Scarlett",
                "Andrew",
                "Madison",
                "Nathan",
                "Avery",
                "Samuel",
                "Lily",
                "Benjamin",
                "Grace",
                "Jack",
                "Chloe",
                "Ryan",
                "Zoe",
                "Jacob",
                "Victoria",
                "Logan",
                "Penelope",
                "Lucas",
                "Riley",
                "Mason",
                "Layla",
                "Ethan",
                "Nora",
            ]

            # List of last_names
            last_names = [
                "Smith",
                "Johnson",
                "Brown",
                "Williams",
                "Jones",
                "Miller",
                "Davis",
                "Garcia",
                "Rodriguez",
                "Martinez",
                "Hernandez",
                "Lopez",
                "Gonzalez",
                "Wilson",
                "Anderson",
                "Thomas",
                "Taylor",
                "Moore",
                "Jackson",
                "Martin",
                "Lee",
                "Perez",
                "Thompson",
                "White",
                "Harris",
                "Clark",
                "Lewis",
                "Walker",
                "Hall",
                "Allen",
                "Young",
                "King",
                "Wright",
                "Scott",
                "Green",
                "Adams",
                "Baker",
                "Nelson",
                "Carter",
                "Mitchell",
                "Perez",
                "Roberts",
                "Phillips",
                "Evans",
                "Campbell",
                "Edwards",
                "Collins",
                "Stewart",
                "Sanchez",
                "Morris",
            ]

            # List of city
            city = [
                "New York",
                "Los Angeles",
                "Chicago",
                "Houston",
                "Phoenix",
                "Philadelphia",
                "San Antonio",
                "San Diego",
                "Dallas",
                "San Jose",
                "Austin",
                "Jacksonville",
                "Fort Worth",
                "Columbus",
                "San Francisco",
                "Charlotte",
                "Indianapolis",
                "Seattle",
                "Denver",
                "Washington, D.C.",
                "Boston",
                "El Paso",
                "Nashville",
                "Detroit",
                "Oklahoma City",
                "Portland",
                "Las Vegas",
                "Memphis",
                "Louisville",
                "Baltimore",
                "Milwaukee",
                "Albuquerque",
                "Tucson",
                "Fresno",
                "Sacramento",
                "Mesa",
                "Kansas City",
                "Atlanta",
                "Omaha",
                "Colorado Springs",
                "Raleigh",
                "Long Beach",
                "Virginia Beach",
                "Miami",
                "Oakland",
                "Minneapolis",
                "Tulsa",
                "Bakersfield",
                "Wichita",
                "Arlington",
            ]

            # Create a dataset with 100 data points
            data = pd.DataFrame(
                {
                    "ID_DF": range(1, 101),
                    "FULLNAME": [
                        f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
                        for _ in range(100)
                    ],
                    "AGE": np.random.randint(20, 60, 100),  # Random age from 20 to 60
                    "SEX": np.random.choice(["Male", "Female"], 100),  # Random SEX
                    "RACE": np.random.choice(
                        ["Asian", "White", "Black", "Latino", "Others"], 100
                    ),
                    "CITY": np.random.choice(city, 100),
                    "Education": np.random.choice(
                        ["High School", "Bachelor", "Master"], 100
                    ),  # Random education level
                    "Income": np.random.randint(
                        30000, 100000, 100
                    ),  # Random income from 30,000 to 100,000
                }
            )
        else:
            data = self.read_data_from_csv(file_path)

        # Add ID_DF column if it does not exist in data
        if "ID_DF" not in data.columns:
            data.insert(0, "ID_DF", range(1, len(data) + 1))

        # Check and adjust invalid values
        for col in data.columns:
            if (
                data[col].dtype == "object"
            ):  # If the column is string replace None values with empty
                data[col] = data[col].replace({None: ""})
            elif np.issubdtype(
                data[col].dtype, np.number
            ):  # If the column is numeric, replace None or nan values with 0
                data[col] = data[col].replace({None: 0, np.nan: 0})

        return data

    def create_data_test_mia(self, data):
        """
        Create data to test functions of Membership Inference Attack
        """

        data_test = data.copy()
        # Check and adjust invalid values
        for col in data_test.columns:
            if (
                data_test[col].dtype == "object"
            ):  # If the column is string replace None values with empty
                data_test[col] = data_test[col].replace({None: ""})
            elif np.issubdtype(
                data_test[col].dtype, np.number
            ):  # If the column is numeric, replace None or nan values with 0
                data_test[col] = data_test[col].replace({None: 0, np.nan: 0})

        # Randomly select 70% of the data_test sample as data_train
        data_train = data_test.sample(frac=0.7, random_state=42)

        data_test["FULLNAME"] = self.generate_name_abbreviations(data_test["FULLNAME"])
        data_test["CITY"] = self.generate_city_abbreviations(data_test["CITY"])

        return data_train, data_test

    def create_data_test_linkage_attack(self, data=None):
        """
        Create data to test functions of Linkage Attack
        """

        if data is None:
            # First dataset
            df1 = pd.DataFrame(
                {
                    "ID_DF1": [1, 2, 3, 4, 5],
                    "FULLNAME": [
                        "John Doe",
                        "Alice Smith",
                        "Bob Brown",
                        "Charlie Johnson",
                        "David Lee",
                    ],
                    "AGE": [25, 30, 35, 40, 45],
                    "SEX": ["Male", "Female", "Male", "Female", "Female"],
                    "RACE": ["Asian", "White", "Black", "Latino", "Others"],
                    "CITY": [
                        "New York",
                        "Los Angeles",
                        "Chicago",
                        "Houston",
                        "Philadelphia",
                    ],
                }
            ).set_index("ID_DF1")

            # Second dataset
            df2 = pd.DataFrame(
                {
                    "ID_DF2": ["A", "B", "C", "D", "E"],
                    "FULLNAME": [
                        "J. Doe",
                        "A. Smith",
                        "B. Brown",
                        "C. Johnson",
                        "D. Lee",
                    ],  # Acronym
                    "AGE": [25, 30, 35, 40, 45],  # Age match
                    "SEX": ["Male", "Female", "Male", "Female", "Female"],
                    "RACE": ["Asian", "White", "Black", "Latino", "Others"],
                    "CITY": ["NY", "LA", "CHI", "HOU", "PHI"],  # Acronym
                    "Income": [2000, 4000, 5000, 3900, 4800],
                    "Education": [
                        "High School",
                        "Bachelor",
                        "Master",
                        "Bachelor",
                        "Master",
                    ],
                }
            ).set_index("ID_DF2")

            return df1, df2
        else:
            data_test = data.copy()

            # Randomly select 70% of the data_test sample as data_train
            data_train = data_test.sample(frac=0.7, random_state=42)

            data_train = data_test.rename(columns={"ID_DF": "ID_DF1"})
            data_test = data_test.rename(columns={"ID_DF": "ID_DF2"})
            data_test["ID_DF2"] = data_test["ID_DF2"].astype(str) + "A"
            data_test["FULLNAME"] = self.generate_name_abbreviations(
                data_test["FULLNAME"]
            )
            data_test["CITY"] = self.generate_city_abbreviations(data_test["CITY"])
            data_test = data_test.set_index("ID_DF2")
            data_train = data_train.set_index("ID_DF1")

            return data_train, data_test

    def display_membership_inference_attack_results(
        self, actual_values, prediction_values
    ):
        """
        Display the results of Membership Inference Attack on the screen
        """

        print("Actual | Prediction | Result\n" + "-" * 30 + "\n")
        for actual, prediction in zip(actual_values, prediction_values):
            result = "✅ True" if actual == prediction else "❌ False"
            print(f"{actual:^7} | {prediction:^7} | {result}")

        # Attack Assessment
        attack_metrics_values = self.attack_metrics.attack_metrics(
            actual_values, prediction_values
        )
        print("\nAttack Metrics Values:")
        for metric, value in attack_metrics_values.items():
            print(f"{metric}: {value}")

        print(
            f"\nAttack Success Rate (ASR): {self.attack_metrics.attack_success_rate(actual_values, prediction_values)}"
        )
        print(
            f"\nResidual Risk Score (RRS): {self.attack_metrics.residual_risk_score(actual_values, prediction_values)}"
        )

    def save_membership_inference_attack_results(
        self, actual_values, prediction_values, file_path
    ):
        """
        Save the result of Membership Inference Attack to file
        """

        attack_metrics_values = self.attack_metrics.attack_metrics(
            actual_values, prediction_values
        )
        asr = self.attack_metrics.attack_success_rate(actual_values, prediction_values)
        rrs = self.attack_metrics.residual_risk_score(actual_values, prediction_values)

        # Check if file_path has a parent directory then create a new one if the directory does not exist
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Write results to file
        with open(file_path, "w", encoding="utf-8") as file:
            header = (
                "\nMembership Inference Attack results:\nActual | Prediction | Result\n"
                + "-" * 30
                + "\n"
            )
            print(header, end="")  # Print to screen
            file.write(header)  # Write to file

            # Compare actual values and predicted values
            for actual, prediction in zip(actual_values, prediction_values):
                result = "✅ True" if actual == prediction else "❌ False"
                line = f"{actual:^7} | {prediction:^7} | {result}\n"
                print(line, end="")  # Print to screen
                file.write(line)  # Write to file

            # Attack Assessment
            metrics_section = "\nAttack Metrics Values:\n"
            print(metrics_section, end="")  # Print to screen
            file.write(metrics_section)  # Write to file

            for metric, value in attack_metrics_values.items():
                metric_line = f"{metric}: {value}\n"
                print(metric_line, end="")
                file.write(metric_line)

            asr_line = f"\nAttack Success Rate (ASR): {asr}\n"
            rrs_line = f"\nResidual Risk Score (RRS): {rrs}\n"

            print(asr_line, end="")  # Print to screen
            print(rrs_line, end="")  # Print to screen
            file.write(asr_line)  # Write to file
            file.write(rrs_line)  # Write to file

        print(f"\n✅ The result has been saved {file_path}")

    def display_attribute_inference_attack_results(
        self, actual_values, prediction_values
    ):
        """
        Display the results of Attribute Inference Attack on the screen
        """

        print("Actual | Prediction | Result\n" + "-" * 30 + "\n")
        for actual, prediction in zip(actual_values, prediction_values):
            result = "✅ True" if actual == prediction else "❌ False"
            print(f"{actual:^7} | {prediction:^7} | {result}")

    def save_attribute_inference_attack_results(
        self, actual_values, prediction_values, file_path
    ):
        """
        Save the results of Attribute Inference Attack to file
        """
        # Check if file_path has a parent directory then create a new one if the directory does not exist
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Write results to file
        with open(file_path, "w", encoding="utf-8") as file:
            header = (
                "\nAttribute Inference Attack results:\nActual | Prediction | Result\n"
                + "-" * 30
                + "\n"
            )
            print(header, end="")
            file.write(header)

            for actual, prediction in zip(actual_values, prediction_values):
                result = "✅ True" if actual == prediction else "❌ False"
                line = f"{actual:^7} | {prediction:^7} | {result}\n"

                print(line, end="")
                file.write(line)

        print(f"\n The result has been saved {file_path}")

    def save_linkage_attack_results(self, results, file_path, title):
        """
        Save the result of a Linkage Attack to file.
        Automatically handles different result formats from the linkage attack methods.
        """

        import pandas as pd
        import os

        # --- Validate the input type ---
        if not isinstance(results, pd.DataFrame):
            raise TypeError(
                f"Expected 'results' to be a pandas DataFrame, got {type(results)}"
            )

        # --- Normalize column names and structure ---
        cols = list(results.columns)

        # Case A: probabilistic_linkage_attack -> ['level_0', 'level_1', 'similarity_score']
        if {"level_0", "level_1"}.issubset(cols):
            results = results.rename(
                columns={
                    "level_0": "ID_DF1",
                    "level_1": "ID_DF2",
                    "similarity_score": "Score",
                }
            )

        # Case B: record_linkage_attack -> merged results with suffixes (_data1, _data2)
        elif any(c.endswith("_data1") for c in cols) or any(
            c.endswith("_data2") for c in cols
        ):
            results = results.reset_index(drop=True).copy()
            results["ID_DF1"] = range(1, len(results) + 1)
            results["ID_DF2"] = range(1, len(results) + 1)
            results["Score"] = None  # No similarity score available

        # Case C: cluster_vector_linkage_attack -> already formatted correctly
        elif not {"ID_DF1", "ID_DF2"}.issubset(results.columns):
            raise ValueError(f"Unsupported result format. Columns found: {cols}")

        # --- Ensure directory exists ---
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # --- Write results to file ---
        with open(file_path, "w", encoding="utf-8") as file:
            header = f"\n\n{title}:\nID_DF1  |  ID_DF2  |  Score\n" + "-" * 30 + "\n"
            print(header, end="")
            file.write(header)

            for row in results.itertuples(index=False):
                score = ""
                if hasattr(row, "Score") and row.Score is not None:
                    score = (
                        f"{row.Score:.6f}"
                        if isinstance(row.Score, (int, float))
                        else str(row.Score)
                    )
                line = f"{getattr(row, 'ID_DF1', ''):^7} | {getattr(row, 'ID_DF2', ''):^7} | {score}\n"
                print(line, end="")
                file.write(line)

        print(f"\n The result has been saved: {file_path}")

    def run_test_membership_inference_attack_dcr(self, data_train, data_test):
        # Create array actual values. Check if each sample in data_test is in data_train? If yes is 1 (member), otherwise 0 (non-member)
        actual_values = np.isin(data_test.index, data_train.index).astype(int)

        # Call the membership_inference_attack_dcr function and get return value as a list containing the inferred values of data_test
        prediction_values = self.membership_inference.membership_inference_attack_dcr(
            data_train, data_test
        )
        self.save_membership_inference_attack_results(
            actual_values, prediction_values, "pamola_core/attacks/results/mia_dcr.txt"
        )

    def run_test_membership_inference_attack_nndr(self, data_train, data_test):
        # Create array actual values. Check if each sample in data_test is in data_train? If yes is 1 (member), otherwise 0 (non-member)
        actual_values = np.isin(data_test.index, data_train.index).astype(int)

        # Call the membership_inference_attack_nndr function and get return value as a list containing the inferred values of data_test
        prediction_values = self.membership_inference.membership_inference_attack_nndr(
            data_train, data_test
        )
        self.save_membership_inference_attack_results(
            actual_values, prediction_values, "pamola_core/attacks/results/mia_nndr.txt"
        )

    def run_test_membership_inference_attack_model(self, data_train, data_test):
        # Create array actual values. Check if each sample in data_test is in data_train? If yes is 1 (member), otherwise 0 (non-member)
        actual_values = np.isin(data_test.index, data_train.index).astype(int)

        # Call the membership_inference_attack_model function and get return value as a list containing the inferred values of data_test
        prediction_values = self.membership_inference.membership_inference_attack_model(
            data_train, data_test
        )
        self.save_membership_inference_attack_results(
            actual_values,
            prediction_values,
            "pamola_core/attacks/results/mia_model.txt",
        )

    def run_test_attribute_inference_attack(self, data_train, data_test):
        # Get the "State" column from data_test, this is the actual value of the attribute to be inferred
        actual_values = data_test["RACE"]
        # Remove "State" column in data_test to make test dataset
        data_test_not_target = data_test.drop(columns="RACE")

        # Call the attribute_inference_attack function and get the return value as a list containing the inferred values of the "State" column of data_test
        prediction_values = self.attribute_inference.attribute_inference_attack(
            data_train, data_test_not_target, "RACE"
        )
        self.save_attribute_inference_attack_results(
            actual_values,
            prediction_values,
            "pamola_core/attacks/results/attribute_inference_attack.txt",
        )

    def run_test_record_linkage_attack(self, data1, data2):
        # Call the function recover_linkage_attack and get the return value as a list of matching record pairs of the two datasets
        results = self.linkage_attack.record_linkage_attack(
            data1, data2, linkage_keys=["AGE", "SEX", "RACE"]
        )
        self.save_linkage_attack_results(
            results,
            "pamola_core/attacks/results/record_linkage_attack.txt",
            "Record Linkage Attack results",
        )

    def run_test_probabilistic_linkage_attack(self, data1, data2):
        # Call the probabilistic_linkage_attack function and get the return value as a list contains pairs of records that match the Fellegi-Sunter score
        results = self.linkage_attack.probabilistic_linkage_attack(
            data1, data2, keys=["FULLNAME", "AGE", "SEX", "RACE", "CITY"]
        )
        self.save_linkage_attack_results(
            results,
            "pamola_core/attacks/results/probabilistic_linkage_attack.txt",
            "Probabilistic Linkage Attack results",
        )

    def run_test_cluster_vector_linkage_attack(self, data1, data2):
        # Call the probabilistic_linkage_attack function and get the return value as a list contains pairs of records that match the PCA & Cosine Similarity score
        results = self.linkage_attack.cluster_vector_linkage_attack(data1, data2)
        self.save_linkage_attack_results(
            results,
            "pamola_core/attacks/results/cluster_vector_linkage_attack.txt",
            "Cluster Vector Linkage Attack results",
        )

    def run_test(self):
        data = self.generate_data()
        # data = self.generate_data("attacks/data_files/RandomPeople_5.csv")
        data_train, data_test = self.create_data_test_mia(data)

        # Test Attribute Inference Attack functions
        self.run_test_attribute_inference_attack(data_train, data_test)

        # Test Membership Inference Attack functions
        self.run_test_membership_inference_attack_dcr(data_train, data_test)
        self.run_test_membership_inference_attack_nndr(data_train, data_test)
        self.run_test_membership_inference_attack_model(data_train, data_test)

        # Test Linkage Attack functions
        data_train, data_test = self.create_data_test_linkage_attack(data)

        self.run_test_record_linkage_attack(data_train, data_test)
        self.run_test_probabilistic_linkage_attack(data_train, data_test)
        self.run_test_cluster_vector_linkage_attack(data_train, data_test)


### Initialize objects
attribute_inference = AttributeInference()
membership_inference = MembershipInference(
    dcr_threshold=None, nndr_threshold=None, m_threshold=None
)
linkage_attack = LinkageAttack(fs_threshold=None, n_components=4)
attack_metrics = AttackMetrics()

### Initialize AttacksTest object
attacks_test = AttacksTest(
    attribute_inference, membership_inference, linkage_attack, attack_metrics
)

attacks_test.run_test()
