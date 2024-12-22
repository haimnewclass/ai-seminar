import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def generate_three_digit_numbers(num_samples=100000):
    """
    יוצר מספרים תלת-ספרתיים רנדומליים ומסמן אם הם זוגיים
    """
    # יצירת מספרים תלת ספרתיים (100-999)
    numbers = torch.randint(100, 1000, (num_samples,))

    # יצירת מערך תוויות (0 לזוגי, 1 לאי-זוגי)
    labels = (numbers % 2 != 0).float()

    # המרת כל מספר למערך של 3 ספרות
    digits = torch.zeros((num_samples, 3))
    digits[:, 0] = numbers // 100  # ספרה ראשונה
    digits[:, 1] = (numbers % 100) // 10  # ספרה שנייה
    digits[:, 2] = numbers % 10  # ספרה שלישית

    # נרמול הספרות לטווח [0,1]
    digits = digits / 9.0

    return digits, labels, numbers


class ThreeDigitClassifier(nn.Module):
    def __init__(self):
        super(ThreeDigitClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 16),  # שכבת קלט: 3 ספרות
            nn.ReLU(),
            nn.Linear(16, 8),  # שכבה נסתרת
            nn.ReLU(),
            nn.Linear(8, 1),  # שכבת פלט: 0/1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def train_model(model, train_digits, train_labels, test_digits, test_labels,
                epochs=1000, batch_size=64, learning_rate=0.001, min_loss=1e-6):
    """
    אימון המודל עם מעקב אחרי ביצועים
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_accuracies = []

    n_batches = len(train_digits) // batch_size

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # ערבוב הדאטה
        indices = torch.randperm(len(train_digits))
        train_digits = train_digits[indices]
        train_labels = train_labels[indices]

        # אימון על מיני-באצ'ים
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_digits = train_digits[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = model(batch_digits)
            loss = criterion(outputs, batch_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # חישוב דיוק על סט הבדיקה
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_digits)
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted.squeeze() == test_labels).float().mean()
            test_accuracies.append(accuracy.item())

        train_losses.append(epoch_loss / n_batches)

        average_loss = epoch_loss / n_batches

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {average_loss:.6f}, Test Accuracy: {accuracy:.4f}')

        # בדיקה האם ה-Loss קרוב מספיק לאפס
        if average_loss < min_loss or average_loss == 0 :
            print(f'\nEarly stopping! Loss reached {average_loss:.6f} at epoch {epoch}')
            break

    return train_losses, test_accuracies


# יצירת הדאטה
digits, labels, original_numbers = generate_three_digit_numbers(15000)

# חלוקה לסט אימון ובדיקה
train_size = int(0.8 * len(digits))
train_digits, test_digits = digits[:train_size], digits[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# יצירת והרצת המודל
model = ThreeDigitClassifier()
train_losses, test_accuracies = train_model(model, train_digits, train_labels, test_digits, test_labels)

# ויזואליזציה של תהליך האימון
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

fig = Figure(figsize=(12, 4))
canvas = FigureCanvasAgg(fig)

# תרשים ראשון - Loss
ax1 = fig.add_subplot(121)
ax1.plot(train_losses)
ax1.set_title('Training Loss over Time')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

# תרשים שני - Accuracy
ax2 = fig.add_subplot(122)
ax2.plot(test_accuracies)
ax2.set_title('Test Accuracy over Time')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')

fig.tight_layout()
canvas.draw()


# פונקציה לבדיקת מספרים ספציפיים
def test_specific_numbers(model, numbers):
    model.eval()
    with torch.no_grad():
        for num in numbers:
            # המרת המספר לפורמט המתאים
            digits = torch.tensor([
                num // 100,
                (num % 100) // 10,
                num % 10
            ], dtype=torch.float32).reshape(1, 3) / 9.0

            output = model(digits)
            predicted_value = output.squeeze().numpy()  # המרה ל-numpy ואז למספר בודד
            predicted = "Odd" if predicted_value > 0.5 else "Even"
            actual = "Odd" if num % 2 else "Even"
            print(f"Number: {num}, Predicted: {predicted}, Actual: {actual}, Raw Output: {predicted_value:.4f}")


# בדיקת המודל על מספרים ספציפיים
test_numbers = [123, 456, 789, 246, 135, 802, 517, 664, 999]
print("\nTesting specific numbers:")
test_specific_numbers(model, test_numbers)