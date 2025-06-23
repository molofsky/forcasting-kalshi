import pandas as pd
import matplotlib.pyplot as plt

def plot_kalshi_data(csv_file):
    # Load dataset
    df = pd.read_csv(csv_file, header=None)

    # Assign column names based on the data schema
    df.columns = ['id', 'market_question', 'date', 'price', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 
                  'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 
                  'col19', 'col20', 'col21', 'col22', 'col23']

    # Filter for the specific market question
    market_question = "Will Trump win the Electoral College and Trump win the Popular Vote?"
    filtered_df = df[df['market_question'] == market_question]

    # Ensure the 'date' column is in datetime format
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])

    # Cut off the data at the specified date
    cutoff_date = pd.to_datetime("2024-11-22")
    filtered_df = filtered_df[filtered_df['date'] <= cutoff_date]

    # Smooth the price data using a rolling mean
    filtered_df['smoothed_price'] = filtered_df['price'].rolling(window=3).mean()  # Adjust window size for smoothing

    # Plotting
    plt.figure(figsize=(10, 5))  # Adjusted figure size
    plt.plot(filtered_df['date'], filtered_df['smoothed_price'], marker='o', linestyle='-', color='b')
    plt.title(market_question, fontsize=16)  # Increased title font size
    plt.xlabel('Date', fontsize=14)  # Increased x-axis label font size
    plt.ylabel('Price', fontsize=14)  # Increased y-axis label font size
    plt.xticks(rotation=45, fontsize=12)  # Increased x-axis tick font size
    plt.yticks(fontsize=12)  # Increased y-axis tick font size
    plt.grid()
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig("trump.png")  # Save the plot with a specified filename
    plt.show()  # Show the plot

if __name__ == "__main__":
    plot_kalshi_data('kalshi.csv')