import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the data
data = {
    'Course': [
        'BioInf - pietro lio',
        'PrincComm - jon crowcroft',
        'InfoTheory - robert harle',
        'MLBI - sean holden',
        'OptComp - timothy jones',
        'Quantum - steven herbert',
        'RandAlg - thomas sauerwald',
        'Hoare - chris pulte',
        'Crypto - martin kleppman'
    ],
    '2024_2': [15, 11.5, 12, 8, 13, 14, 14.5, 13, 14],
    '2024_1': [14, 17, 13, 9.5, 13, 15, 16, 14, 10],
    '2023_2': [14, 15, 11, 15.5, 14, 10, 14, 14, 13],
    '2023_1': [15, 13, 14, 10, 11, 10, 15, 15.5, 12],
    '2022_2': [14, 14, 14, 14, 12, 13, 15, 16, 11],
    '2022_1': [12, 14, 13.5, 15, 12, 9, 14, 15, 12]
}

# Create DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to create a long format suitable for time series
df_melted = pd.melt(df, 
                    id_vars=['Course'], 
                    value_vars=['2022_1', '2022_2', '2023_1', '2023_2', '2024_1', '2024_2'],
                    var_name='Period',
                    value_name='Score')

# Create the plot
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Create line plot
for course in df['Course']:
    course_data = df_melted[df_melted['Course'] == course]
    plt.plot(course_data['Period'], course_data['Score'], marker='o', label=course)

# Customize the plot
plt.title('Course Scores Over Time (2022-2024)', pad=20)
plt.xlabel('Time Period')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()