import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback

class WeightsLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}")
        for layer in self.model.layers:
            weights, biases = layer.get_weights()
            print(f"Layer: {layer.name}")
            print(f"Weights:\n{weights}")
            print(f"Biases:\n{biases}")

data = {
    'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
    'feature2': [0.5, 0.4, 0.3, 0.2, 0.1],
    'label': [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
print(df)

x = df[['feature1', 'feature2']].values
print(x)

y = df['label'].values
print(y)

#Initialization of layers
#Typically created by using Dense layers
model1 = Sequential([
    Dense(units=8, activation='relu', input_dim=2),
    Dense(units=1, activation='sigmoid'),
])

#Compiling the model by specifying the loss function, optimizer and metrics to evaluate during training
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Training the model
model1.fit(x, y, epochs=10, batch_size=1, verbose=1, callbacks=[WeightsLogger()])

#Now we use test data to make predictions
test_data = np.array([[0.2, 0.4]])
prediction = model1.predict(test_data)
predicted_label = (prediction > 0.5).astype(int)

print(prediction)
print(predicted_label)


