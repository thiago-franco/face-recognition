from functions import *
import sys

def main():
    action = sys.argv[1:][0]
    path = sys.argv[1:][1] 
    if(action == 'eval'):
        evaluate_recognition(path, LogisticRegression())
    elif(action == 'rec'):
        X, y = load_dataset(path)
        pca = PCA()
        X = pca.fit_transform(X)
        model, score, cm = classify(X, y)
        print('Recognition Rate: %s' % score)
        print('Confuision Matrix:')
        print(cm)
        images_path = sys.argv[1:][2]
        files = get_file_names(images_path)
        correct = 0
        for file in files:
            act, pred = recognize_image(file, model, pca)
            if act == pred:
                correct += 1
            print('Face %s recognized as: %s' % (get_image_name(file), pred))
        print("%s (%s/%s) correctly classified." % (correct/len(files), correct, len(files)))

main()