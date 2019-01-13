from functions import *
import sys

def main():
    action = sys.argv[1:][0]
    path = sys.argv[1:][1] 
    if(action == 'eval'):
        evaluate_recognition(path, LogisticRegression())
    elif(action == 'rec'):
        image_to_rec = sys.argv[1:][2]
        X, y = load_dataset(path)
        pca = PCA()
        X = pca.fit_transform(X)
        model, score, cm = classify(X, y)
        print('Recognition Rate: %s' % score)
        print('Confuision Matrix:')
        print(cm)
        act, pred = recognize_image(image_to_rec, model, pca)
        print('Face recognized as: %s' % pred)

main()