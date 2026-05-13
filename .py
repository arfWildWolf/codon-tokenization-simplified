import matplotlib.pyplot as plt
import numpy as np

# Data: [Name, X (Ease of Implementation), Y (Education Specificity), Color Group]
# X-axis: 0 (Hard/Expensive) to 10 (Easy/Hardware-free)
# Y-axis: 0 (General Corporate) to 10 (Highly Education-focused)

competitors = [
    # The Leader
    ["UniDay", 9.5, 9.5, "blue"],
    
    # General Attendance / SaaS (High Ease, Low Ed-Specificity)
    ["Jibble", 8.5, 4.0, "green"],
    ["Clockify", 9.0, 3.5, "green"],
    ["Homebase", 8.0, 3.0, "green"],
    ["Buddy Punch", 7.5, 3.8, "green"],
    ["Deputy", 7.0, 3.2, "green"],
    ["QuickBooks Time", 6.5, 2.5, "green"],

    # Enterprise HRMS (Low Ease/High Complexity, Low Ed-Specificity)
    ["Workday", 2.0, 2.0, "red"],
    ["SAP SuccessFactors", 1.5, 1.8, "red"],
    ["Zoho People", 4.5, 3.5, "red"],
    ["BambooHR", 5.0, 3.0, "red"],
    ["Gusto", 6.0, 2.0, "red"],

    # Education/LMS Tools (Moderate Ease, High Ed-Specificity)
    ["Canvas LMS", 6.0, 8.5, "orange"],
    ["Moodle", 5.0, 8.0, "orange"],
    ["Blackboard", 4.0, 8.2, "orange"],
    ["Top Hat", 7.0, 8.8, "orange"],

    # Hardware & Traditional (Low Ease, Low/Mid Utility)
    ["ZKTeco (Hardware)", 2.0, 4.0, "gray"],
    ["HID Global", 1.5, 4.5, "gray"],
    ["Paper Sign-in", 3.0, 5.0, "gray"],
    ["Excel Sheets", 4.0, 4.0, "gray"]
]

# Extract data
names, x, y, colors = zip(*competitors)

plt.figure(figsize=(14, 10))
plt.scatter(x, y, c=colors, s=150, alpha=0.7, edgecolors='w')

# Annotate names
for i, txt in enumerate(names):
    font_weight = 'bold' if txt == "UniDay" else 'normal'
    plt.annotate(txt, (x[i], y[i]), xytext=(7, 7), textcoords='offset points', fontsize=10, weight=font_weight)

# Draw Quadrant Lines
plt.axvline(x=5, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(y=5, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Labels and Styling
plt.title("Competitive Analysis: Attendance Systems for Education", fontsize=16, pad=20)
plt.xlabel("Ease of Implementation (Cost & Hardware Accessibility) →", fontsize=12)
plt.ylabel("Education Specificity (Classroom Workflow & Features) →", fontsize=12)

# Quadrant Labels
plt.text(1, 9.5, "STRATEGIC EDUCATION TOOLS", fontsize=12, color='gray', alpha=0.5)
plt.text(7.5, 9.5, "MARKET LEADERS", fontsize=12, color='blue', alpha=0.5, weight='bold')
plt.text(1, 0.5, "TRADITIONAL / ENTERPRISE", fontsize=12, color='gray', alpha=0.5)
plt.text(7.5, 0.5, "GENERAL NICHE", fontsize=12, color='gray', alpha=0.5)

plt.grid(True, which='both', linestyle=':', alpha=0.3)
plt.xlim(0, 11)
plt.ylim(0, 11)
plt.tight_layout()
plt.show()