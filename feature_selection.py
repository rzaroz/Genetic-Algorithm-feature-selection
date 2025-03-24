import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class GeneticEngine:
    def __init__(self, x, y, target, k):
        self.X = x
        self.y = y
        self.k = k
        self.genes = list(x.columns)
        self.target = target

        if len(self.genes) <= self.target:
            raise ValueError("The target value must be smaller than the number of features.")

        self.parent = None
        self.parentAccuracy = None

    def generate_starter_features(self):
        guess = random.sample(self.genes, self.target)
        return guess

    def fitness(self, guess):
        childX = self.X[guess]

        X_train, X_test, y_train, y_test = train_test_split(childX, self.y,  test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_hat = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_hat)

        return accuracy

    def mutate(self, guess):
        child_genes = list(set(self.genes) - set(self.parent))
        gen = random.sample(child_genes, self.k)
        indices = random.sample(range(len(guess)), len(gen))

        for i in range(len(gen)):
            guess[indices[i]] = gen[i]

        return guess

    def display(self, acc, counter):
        print(f"try number: {counter} | Accuracy: {acc}")

    def fit(self):
        self.parent = self.generate_starter_features()
        self.parentAccuracy = self.fitness(self.parent)
        self.display(self.parentAccuracy, 'First guess')

        counter = 1
        while True:
            counter += 1

            if counter == 100:
                stop_access = str(input("Can i stop? y or n: "))
                if stop_access == 'y':
                    break
                else:
                    counter = 1

            child = self.mutate(self.parent)
            childFit = self.fitness(child)

            if childFit <= self.parentAccuracy:
                continue

            self.parent = child
            self.parentAccuracy = childFit
            self.display(self.parentAccuracy, counter)


df = pd.read_csv("BreastCancerWisconsin(Diagnostic).csv")
X = df.drop(["id", "diagnosis"], axis=1)
y = LabelEncoder().fit_transform(df["diagnosis"])

engine = GeneticEngine(X, y, 5, 2)
engine.fit()
print(f"Final features: {engine.parent}")