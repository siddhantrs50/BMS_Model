def recommend_action(row):
    if row['Predicted Status'] == 'Critical':
        return "Immediate servicing. Avoid high speeds and long trips."
    elif row['Predicted Status'] == 'Warning':
        if row['Max Cell Temp'] > 60:
            return "Reduce driving speed to lower thermal stress."
        elif row['Internal Resistance'] > 0.004:
            return "Consider battery check-up. Avoid deep discharges."
        else:
            return "Optimize charge cycles. Maintain 20–80% range."
    else:
        return "Battery is healthy. No action needed."
