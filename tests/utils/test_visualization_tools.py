import unittest
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from core.utils.visualization_tools import get_risk_color_map


class TestGetRiskColorMap(unittest.TestCase):

    def test_get_risk_color_map(self):
        expected_color_map = {
            'VERY LOW': '#4CAF50',  # Green
            'LOW': '#8BC34A',  # Light Green
            'MODERATE': '#FFC107',  # Amber
            'HIGH': '#FF9800',  # Orange
            'VERY HIGH': '#F44336'  # Red
        }

        # Call the function
        actual_color_map = get_risk_color_map()

        # Check if the actual color map matches the expected color map
        self.assertEqual(actual_color_map, expected_color_map)

    def test_risk_level_bar_chart(self):
        # Assume we have some risk levels
        risk_levels = ['VERY LOW', 'LOW', 'MODERATE', 'HIGH', 'VERY HIGH']
        color_map = get_risk_color_map()

        # Create a bar chart to visualize risk levels
        plt.figure()  # Create a new figure
        plt.bar(risk_levels, [1, 2, 3, 4, 5], color=[color_map[level] for level in risk_levels])
        plt.title('Risk Levels Visualization')
        plt.xlabel('Risk Level')
        plt.ylabel('Value')

        # Show the plot
        # plt.show()  # This will display the plot window

        # Check if the figure has been created
        self.assertIsNotNone(plt.gcf())  # Get current figure and check it's not None

        # Check if the number of bars is correct
        bars = plt.gca().patches  # Get the bars from the current axes
        self.assertEqual(len(bars), len(risk_levels))  # Number of bars should match risk levels

        # Check if the colors of the bars are correct
        for bar, level in zip(bars, risk_levels):
            expected_color = color_map[level]
            # Convert the expected color to RGBA format for comparison
            self.assertEqual(bar.get_facecolor(), mcolors.to_rgba(expected_color))  # Check color

        plt.close()  # Close the figure after testing


if __name__ == '__main__':
    unittest.main()