#include <iostream>
#include <fstream>
using namespace std;

int main() {
	//ifstream in_test("/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/test_unorder.txt");
	//ifstream in_train("/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/train_unorder.txt");
	ifstream in("/media/lvshq/LSQ_16GB/SSDH/pic_for_windows.txt");

	ofstream out1("/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/pic_for_windows_with_prefix.txt");

	string name, prefix;
	int label;
	prefix = "G:/Software/caffe-windows/caffe-master/data/Clothes/dataset/";
	if (in.is_open() && out1.is_open()) {
		while (in >> name) {
			out1 << prefix << name << endl;
		}
	} else {
		cout << "Train not opened" << endl;
	}
	in.close();
	out1.close();



/*
	if (in_train.is_open() && in_test.is_open() && out1.is_open()) {
		while (in_train >> name >> label) {
			out1 << prefix << name << endl;
		}
		while (in_test >> name >> label) {
			out1 << prefix << name << endl;
		}
	} else {
		cout << "Train not opened" << endl;
	}
	in_test.close();
	in_train.close();
	out1.close();
*/
	/*
	ifstream in_test("/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/test_unorder.txt");
	ifstream in_train("/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/train_unorder.txt");

	ofstream out_test("/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/test-file-list.txt");
	ofstream out_test_label("/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/test-label.txt");
	ofstream out_train("/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/train-file-list.txt");
	ofstream out_train_label("/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/train-label.txt");

	string name, prefix;
	string prefix_train = "/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/train_imgs/";
	string prefix_test = "/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/test_imgs/";
	int label;
	if (in_train.is_open() && out_train.is_open() && out_train_label.is_open()) {
		while (in_train >> name >> label) {
			out_train << prefix_train << name << endl;
			out_train_label << label << endl;
		}
	} else {
		cout << "Train not opened" << endl;
	}
	if (in_test.is_open()) {
		while (in_test >> name >> label) {
			out_test << prefix_test << name << endl;
			out_test_label << label << endl;
		}
	} else {
		cout << "Test not opened" << endl;
	}
	*/


	return 0;
}