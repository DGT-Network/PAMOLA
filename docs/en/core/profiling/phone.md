import pandas as pd
from pathlib import Path
from pamola_core.profiling.analyzers.phone import PhoneOperation

# Create custom messenger patterns CSV
patterns_df = pd.DataFrame([
    {'messenger_type': 'telegram', 'pattern': 'tg_account'},
    {'messenger_type': 'whatsapp', 'pattern': 'вацап'},
    {'messenger_type': 'discord', 'pattern': 'дискорд'}
])
patterns_csv = Path("./custom_messenger_patterns.csv")
patterns_df.to_csv(patterns_csv, index=False)

# Create and execute operation with custom patterns
operation = PhoneOperation(
    field_name="cell_phone",
    min_frequency=1,
    patterns_csv=str(patterns_csv)
)

result = operation.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter
)

# Check messenger results
if result.status == OperationStatus.SUCCESS:
    messenger_artifacts = [a for a in result.artifacts if "messenger" in str(a.path)]
    if messenger_artifacts:
        print(f"Messenger analysis saved to: {messenger_artifacts[0].path}")
        
        # You could load and examine the CSV or JSON data if needed
        import json
        with open([a.path for a in messenger_artifacts if a.path.endswith('.json')][0], 'r') as f:
            messenger_data = json.load(f)
            
        print("Messenger mentions found:")
        for messenger in messenger_data['messengers']:
            print(f"  {messenger['messenger']}: {messenger['count']} ({messenger['percentage']}%)")