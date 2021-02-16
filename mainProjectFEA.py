# -*- coding: utf-8 -*-
"""This is a 2D Frame Analysis Programme by using Finite Element Method """
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
from matplotlib import pyplot as plt
import AppInterface
import pandas as pd

class FiniteObject(QtWidgets.QMainWindow):
    """Finite Element Object"""
    def __init__(self):
        super(FiniteObject, self).__init__()
        self.uiMenu = AppInterface.Ui_Form()
        self.uiMenu.setupUi(self)
        self.setFixedSize(855, 360)
        self.uiMenu.fixed_label_2.setPixmap(QtGui.QPixmap("fixed.PNG"))
        self.uiMenu.pinned_label_2.setPixmap(QtGui.QPixmap("pinned.PNG"))
        self.uiMenu.roller_label_2.setPixmap(QtGui.QPixmap("roller.PNG"))
        self.node_count = 1
        self.element_ID = 1
        self.coordx = []    # X koordinatları bu listede depolanacak.
        self.coordy = []    # Y koordinatları bu listede depolanacak.
        self.x_coor_elem = []
        self.y_coor_elem = []
        self.node_coords = []
        self.section_name = []
        self.element_side_left = []
        self.element_side_right = []
        self.element_prop = {}
        self.element_ID_area_inertia_elasticity_length_cos_sin_dict = {}
        self.node_and_type_dict = {}
        self.joint_node_and_loads = {}
        self.distributed_load_and_elements = {}
        self.element_CosSin_values = {}
        self.element_stiffness_matrix_dict = {}
        self.element_transformation_matrix_dict = {}
        self.uiMenu.x_magnitude.setText("0")
        self.uiMenu.y_magnitude.setText("0")
        self.uiMenu.moment_magnitude.setText("0")
        self.uiMenu.distributed_load.setText("0")
        self.uiMenu.save_node_button.clicked.connect(self.save_nodes)
        self.uiMenu.remove_node_button.clicked.connect(self.remove_nodes)
        self.uiMenu.save_element_button.clicked.connect(self.save_elements)
        self.uiMenu.remove_element_button.clicked.connect(self.remove_elements)
        self.uiMenu.assign_element_button.clicked.connect(self.assign_elements)
        self.uiMenu.save_support_button.clicked.connect(self.assign_supports)
        self.uiMenu.pinned_check_2.stateChanged.connect(self.check_support_type)
        self.uiMenu.roller_check_2.stateChanged.connect(self.check_support_type)
        self.uiMenu.fixed_check_2.stateChanged.connect(self.check_support_type)
        self.uiMenu.add_point_load_button.clicked.connect(self.add_point_load)
        self.uiMenu.add_distributed_load_button.clicked.connect(self.add_distributed_load)
        self.uiMenu.analysis_button.clicked.connect(self.analysis_system)
    def save_nodes(self):
        """Save Node Function"""
        x_coor = int(self.uiMenu.node_x.text())
        y_coor = int(self.uiMenu.node_y.text())
        self.coordx.append(x_coor)
        self.coordy.append(y_coor)
        self.node_coords.append([x_coor, y_coor])
        printText = f"node{self.node_count}  |   X:{x_coor}  |   Y:{y_coor}"
        self.uiMenu.node_list.addItem(printText)
        self.uiMenu.left_side_node.addItem(str(self.node_count))
        self.uiMenu.right_side_node.addItem(str(self.node_count))
        self.uiMenu.support_node_box.addItem(str(self.node_count))
        self.uiMenu.node_box.addItem(str(self.node_count))
        self.node_count += 1
        self.node_graph()
    def node_graph(self):
        """Draw Node Graphic Function"""
        plt.figure(figsize=(4, 3))
        plt.scatter(self.coordx, self.coordy)
        for i in range(len(self.coordx)):
            plt.annotate(f"{i+1}({self.coordx[i]}, {self.coordy[i]})", (self.coordx[i], self.coordy[i]), fontsize=10)
        plt.savefig("graph.png", dpi=100)
        self.uiMenu.display_label_node.setPixmap(QtGui.QPixmap("graph.png"))
    def remove_nodes(self):
        """Remove Nodes Function"""
        remove_node_id = self.uiMenu.node_list.currentRow()
        self.coordx.pop(remove_node_id)
        self.coordy.pop(remove_node_id)
        self.node_coords.pop(remove_node_id)
        self.uiMenu.node_list.takeItem(self.uiMenu.node_list.currentRow())
        self.uiMenu.left_side_node.removeItem(remove_node_id)
        self.uiMenu.right_side_node.removeItem(remove_node_id)
        self.uiMenu.support_node_box.removeItem(remove_node_id)
        self.uiMenu.node_box.removeItem(remove_node_id)
        self.node_graph()
    def save_elements(self):
        """Save Elements Function"""
        element_name = self.uiMenu.elem_name.text()
        element_width = float(self.uiMenu.elem_width.text())
        element_height = float(self.uiMenu.elem_height.text())
        elasticity = float(self.uiMenu.elem_elasticity.text())
        element_area = element_width * element_height
        inertia_moment = (element_width * element_height ** 3) / 12
        element_inertia = inertia_moment * float(self.uiMenu.elem_inertia_coeff.text())
        area_notation = f"{element_area:.2E}"
        inertia_notation = f"{element_inertia:.2E}"
        if element_name not in self.section_name:
            self.section_name.append(element_name)
            self.element_prop[element_name] = [element_area, element_inertia, elasticity]
            self.uiMenu.section_list_box.addItem(element_name)
            self.uiMenu.element_list.addItem(f"{element_name} -- A:{str(area_notation)} | I:{str(inertia_notation)}")
        else:
            message = QtWidgets.QMessageBox()
            message.setWindowTitle("Same Section Name")
            message.setText("Please change section name.")
            exit_message = message.exec_()
    def remove_elements(self):
        """Remove Elements Function"""
        self.uiMenu.element_list.takeItem(self.uiMenu.element_list.currentRow())
    def assign_elements(self):
        """Assing Elements Function"""
        self.uiMenu.tabWidget_2.setCurrentIndex(1)
        self.right_node = int(self.uiMenu.right_side_node.currentText()) - 1
        self.left_node = int(self.uiMenu.left_side_node.currentText()) - 1
        self.element_side_left.append(self.left_node)
        self.element_side_right.append(self.right_node)
        element_length = np.sqrt((self.coordx[self.right_node] - self.coordx[self.left_node]) ** 2 + (self.coordy[self.right_node] - self.coordy[self.left_node]) ** 2)
        cos_value = (self.coordx[self.right_node] - self.coordx[self.left_node]) / element_length
        sin_value = (self.coordy[self.right_node] - self.coordy[self.left_node]) / element_length
        self.element_ID_area_inertia_elasticity_length_cos_sin_dict[self.element_ID] = [self.element_prop[self.uiMenu.section_list_box.currentText()][0], self.element_prop[self.uiMenu.section_list_box.currentText()][1], self.element_prop[self.uiMenu.section_list_box.currentText()][2], element_length, cos_value, sin_value]
        self.x_coor_elem.append([self.coordx[self.right_node], self.coordx[self.left_node]])
        self.y_coor_elem.append([self.coordy[self.right_node], self.coordy[self.left_node]])
        message = QtWidgets.QMessageBox()
        message.setWindowTitle("Member Information")
        message.setText(f"Assigned {self.uiMenu.section_list_box.currentText()} member between {self.left_node + 1}. and {self.right_node + 1}. nodes!")
        message_win = message.exec_()
        self.uiMenu.member_box.addItem(str(self.element_ID))
        self.element_ID += 1
        print(self.element_ID_area_inertia_elasticity_length_cos_sin_dict)
        self.element_graph()
    def element_graph(self):
        """Drawing 2D Graph Func"""
        plt.figure(figsize=(4, 3))
        for i in range(len(self.element_side_left)):
            plt.plot([self.node_coords[self.element_side_left[i]][0], self.node_coords[self.element_side_right[i]][0]], [self.node_coords[self.element_side_left[i]][1], self.node_coords[self.element_side_right[i]][1]], marker="o")
        plt.savefig("graph.png", dpi=100)
        self.uiMenu.display_label.setPixmap(QtGui.QPixmap("graph.png"))
    def calculate_system_stiffness_matrix(self):
        """Calculating System Stiffness Matrix"""
        self.system_stiffness_matrix = np.zeros((len(self.coordx) * 3, (len(self.coordx)) * 3))
        self.transformed_stiffness_dict = {}
        for operate in self.element_ID_area_inertia_elasticity_length_cos_sin_dict.keys():
            element_stiffMtx = np.empty((6, 6))
            length = self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(operate)[3]
            elasticity_module = self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(operate)[2]
            inertia_moment = self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(operate)[1]
            section_area = self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(operate)[0]
            element_stiffMtx[0, 0] = elasticity_module * section_area / length
            element_stiffMtx[0, 1] = 0
            element_stiffMtx[0, 2] = 0
            element_stiffMtx[0, 3] = -1 * elasticity_module * section_area / length
            element_stiffMtx[0, 4] = 0
            element_stiffMtx[0, 5] = 0
            element_stiffMtx[1, 0] = 0
            element_stiffMtx[1, 1] = 12 * elasticity_module * inertia_moment / length ** 3
            element_stiffMtx[1, 2] = 6 * elasticity_module * inertia_moment / length ** 2
            element_stiffMtx[1, 3] = 0
            element_stiffMtx[1, 4] = -12 * elasticity_module * inertia_moment / length ** 3
            element_stiffMtx[1, 5] = 6 * elasticity_module * inertia_moment / length ** 2
            element_stiffMtx[2, 0] = 0
            element_stiffMtx[2, 1] = 6 * elasticity_module * inertia_moment / length ** 2
            element_stiffMtx[2, 2] = 4 * elasticity_module * inertia_moment / length
            element_stiffMtx[2, 3] = 0
            element_stiffMtx[2, 4] = -6 * elasticity_module * inertia_moment / length ** 2
            element_stiffMtx[2, 5] = 2 * elasticity_module * inertia_moment / length
            element_stiffMtx[3, 0] = -1 * elasticity_module * section_area / length
            element_stiffMtx[3, 1] = 0
            element_stiffMtx[3, 2] = 0
            element_stiffMtx[3, 3] = elasticity_module * section_area / length
            element_stiffMtx[3, 4] = 0
            element_stiffMtx[3, 5] = 0
            element_stiffMtx[4, 0] = 0
            element_stiffMtx[4, 1] = -12 * elasticity_module * inertia_moment / length ** 3
            element_stiffMtx[4, 2] = -6 * elasticity_module * inertia_moment / length ** 2
            element_stiffMtx[4, 3] = 0
            element_stiffMtx[4, 4] = 12 * elasticity_module * inertia_moment / length ** 3
            element_stiffMtx[4, 5] = -6 * elasticity_module * inertia_moment / length ** 2
            element_stiffMtx[5, 0] = 0
            element_stiffMtx[5, 1] = 6 * elasticity_module * inertia_moment / length ** 2
            element_stiffMtx[5, 2] = 2 * elasticity_module * inertia_moment / length
            element_stiffMtx[5, 3] = 0
            element_stiffMtx[5, 4] = -6 * elasticity_module * inertia_moment / length ** 2
            element_stiffMtx[5, 5] = 4 * elasticity_module * inertia_moment / length
            print(element_stiffMtx)
            self.element_stiffness_matrix_dict[operate] = element_stiffMtx
            # data_element_stiffness = pd.DataFrame(element_stiffMtx)
            # data_element_stiffness.to_excel(f"{operate}. Eleman Rijitlik Matrisi.xlsx")
            # Calculate Transformation Matrix
            self.transformation_matrix(self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(operate)[4], self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(operate)[5])
            print(self.transformationMtrx)
            self.element_transformation_matrix_dict[operate] = self.transformationMtrx
            # data_transfMtrx = pd.DataFrame(self.transformationMtrx)
            # data_transfMtrx.to_excel(f"{operate}. Eleman Transformasyon Matrisi.xlsx")
            self.transfTransp_multiply_stiff = (np.transpose(self.transformationMtrx)).dot(element_stiffMtx)
            self.transformedEleStiffMtrx = (self.transfTransp_multiply_stiff).dot(self.transformationMtrx)
            self.transformed_stiffness_dict[operate] = element_stiffMtx
            # data_roundTranf = pd.DataFrame(self.transformedEleStiffMtrx)
            # data_roundTranf.to_excel(f"{operate}. Eleman Dönüştürülmüş Rijitlik Matrisi.xlsx")
            self.system_stiffness_matrix[(operate - 1) * 3 :(operate - 1) * 3 + 6, (operate - 1) * 3 : (operate - 1) * 3 + 6] += self.transformedEleStiffMtrx
        print(self.system_stiffness_matrix)
        data_stiff_system = pd.DataFrame(self.system_stiffness_matrix)
        data_stiff_system.to_excel("Toplam Sistem Rijitlik Matrisi.xlsx")
    def transformation_matrix(self, cos_value, sin_value):
        """Transformation Matrix Funct"""
        self.transformationMtrx = np.zeros((6, 6))
        self.transformationMtrx[0, 0] = cos_value
        self.transformationMtrx[0, 1] = sin_value
        self.transformationMtrx[1, 0] = sin_value * (-1)
        self.transformationMtrx[1, 1] = cos_value
        self.transformationMtrx[2, 2] = 1
        self.transformationMtrx[3, 3] = cos_value
        self.transformationMtrx[3, 4] = sin_value
        self.transformationMtrx[4, 3] = sin_value * (-1)
        self.transformationMtrx[4, 4] = cos_value
        self.transformationMtrx[5, 5] = 1
    def assign_supports(self):
        """Assign Support Funct"""
        support_node = int(self.uiMenu.support_node_box.currentText()) - 1
        self.node_and_type_dict[support_node] = self.support_text
        print(self.node_and_type_dict)
        message = QtWidgets.QMessageBox()
        message.setWindowTitle("Support Condition")
        message.setText(f"Created a {self.support_text} at {support_node + 1}. node!")
        message_win = message.exec_()
    def check_support_type(self, state):
        """Check Button Controls"""
        if state == QtCore.Qt.Checked:
            if self.sender() == self.uiMenu.pinned_check_2:
                self.support_text = self.uiMenu.pinned_check_2.text()
                self.uiMenu.roller_check_2.setChecked(False)
                self.uiMenu.fixed_check_2.setChecked(False)
            elif self.sender() == self.uiMenu.roller_check_2:
                self.support_text = self.uiMenu.roller_check_2.text()
                self.uiMenu.pinned_check_2.setChecked(False)
                self.uiMenu.fixed_check_2.setChecked(False)
            elif self.sender() == self.uiMenu.fixed_check_2:
                self.support_text = self.uiMenu.fixed_check_2.text()
                self.uiMenu.pinned_check_2.setChecked(False)
                self.uiMenu.roller_check_2.setChecked(False)
    def add_point_load(self):
        """Add Point Load"""
        joint_load_node = int(self.uiMenu.node_box.currentText()) - 1
        self.joint_node_and_loads[joint_load_node] = [float(self.uiMenu.x_magnitude.text()), float(self.uiMenu.y_magnitude.text()), float(self.uiMenu.moment_magnitude.text())]
        message = QtWidgets.QMessageBox()
        message.setWindowTitle("Point Load")
        message.setText(
            f"At the {joint_load_node + 1}. point, a {self.uiMenu.x_magnitude.text()} kN dir X and {self.uiMenu.y_magnitude.text()} kN dir Y and {self.uiMenu.moment_magnitude.text()} kNm moment load was defined!")
        message_win = message.exec_()
    def add_distributed_load(self):
        """Add Distributed Load"""
        distributed_load_element = int(self.uiMenu.member_box.currentText()) - 1
        self.distributed_load_and_elements[distributed_load_element] = float(self.uiMenu.distributed_load.text())
        self.element_load_vector = np.array([[0],
                                     [float(self.uiMenu.distributed_load.text()) * self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(distributed_load_element + 1)[3] / 2],
                                     [float(self.uiMenu.distributed_load.text()) * self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(distributed_load_element + 1)[3] ** 2 / 12],
                                     [0],
                                     [float(self.uiMenu.distributed_load.text()) * self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(distributed_load_element + 1)[3] / 2],
                                     [-1 * float(self.uiMenu.distributed_load.text()) * self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(distributed_load_element + 1)[3] ** 2 / 12]])
        self.transformation_matrix(self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(distributed_load_element + 1)[4], self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(distributed_load_element + 1)[5])
        self.load_vector = np.dot(self.transformationMtrx, self.element_load_vector)
        print("load vector:",self.load_vector)
        print("ELEMlOAD:",self.element_load_vector)
        message = QtWidgets.QMessageBox()
        message.setWindowTitle("Distributed Load")
        message.setText(
            f"At the {distributed_load_element + 1}. member, a {self.uiMenu.distributed_load.text()} kN/m distributed load was defined!")
        message_win = message.exec_()
    def analysis_system(self):
        """Analysis System Funct"""
        self.calculate_system_stiffness_matrix()
        self.load_vector_ofSystem()
        self.system_analysis()
        self.element_displacement()
        self.draw_displacement()
        self.create_report()
        # self.draw_axial_loads()
        # self.draw_shear_forces()
        # self.draw_moment_diagram()
    def load_vector_ofSystem(self):
        """Create System Load Vector"""
        self.system_load_matrix = np.zeros(((len(self.coordx)) * 3, 1))
        for jload in self.joint_node_and_loads.keys():
            self.system_load_matrix[jload * 3, 0] += self.joint_node_and_loads[jload][0]
            self.system_load_matrix[jload * 3 + 1, 0] += self.joint_node_and_loads[jload][1]
            self.system_load_matrix[jload * 3 + 2, 0] += self.joint_node_and_loads[jload][2]
        for dload in self.distributed_load_and_elements.keys():
            self.system_load_matrix[(dload) * 3: (dload) * 3 + 6, 0] += self.load_vector[:, 0]
    def system_analysis(self):
        """System Analysis"""
        self.uiMenu.tabWidget_2.setCurrentIndex(2)
        self.revized_StiffnessList = []
        self.revized_LoadList = []
        for node in self.node_and_type_dict.keys():
            if self.node_and_type_dict[node] == "Pinned Support" :
                self.system_stiffness_matrix[node * 3 : node * 3 + 2, :] = 1.123456789
                self.system_stiffness_matrix[:, node * 3 : node * 3 + 2] = 1.123456789
                self.system_load_matrix[node * 3 : node * 3 + 2, 0] = 1.123456789
            if self.node_and_type_dict[node] == "Roller Support" :
                self.system_stiffness_matrix[node * 3 + 1, :] = 1.123456789
                self.system_stiffness_matrix[:, node * 3 + 1] = 1.123456789
                self.system_load_matrix[node * 3 : node * 3 + 1, 0] = 1.123456789
            if self.node_and_type_dict[node] == "Fixed Support" :
                self.system_stiffness_matrix[node * 3 : node * 3 + 3, :] = 1.123456789
                self.system_stiffness_matrix[:, node * 3 : node * 3 + 3] = 1.123456789
                self.system_load_matrix[node * 3 : node * 3 + 3, 0] = 1.123456789
        for row in self.system_stiffness_matrix:
            for item in row:
                if item != 1.123456789:
                    self.revized_StiffnessList.append(item)
        self.revized_StiffnessMtrx = np.array(self.revized_StiffnessList).reshape(
            (int(len(self.revized_StiffnessList) ** 0.5), int(len(self.revized_StiffnessList) ** 0.5)))
        for row in self.system_load_matrix:
            for item in row:
                if item != 1.123456789:
                    self.revized_LoadList.append(item)
        self.revized_LoadMtrx = np.array(self.revized_LoadList).reshape((int(len(self.revized_LoadList))), 1)
        # data_load = pd.DataFrame(self.revized_LoadMtrx)
        # data_load.to_excel("Yük Vektörü.xlsx")
        data_system = pd.DataFrame(self.revized_StiffnessMtrx)
        data_system.to_excel("Revize Sistem Rijitlik Matrisi.xlsx")
        self.reverse_system_stiffMtrx = np.linalg.inv(self.revized_StiffnessMtrx)
        # data_inverse = pd.DataFrame(self.reverse_system_stiffMtrx)
        # data_inverse.to_excel("inverse.xlsx")
        self.system_displacement_vector = np.dot(self.reverse_system_stiffMtrx, self.revized_LoadMtrx)
        self.displacement_list = []
        for item in self.system_displacement_vector :
            for number in item :
                self.displacement_list.append(number)
        for node in self.node_and_type_dict.keys() :
            if self.node_and_type_dict[node] == "Pinned Support" :
                self.displacement_list.insert(node * 3, 0)
                self.displacement_list.insert(node * 3 + 1, 0)
            if self.node_and_type_dict[node] == "Roller Support" :
                self.displacement_list.insert(node * 3 + 1, 0)
            if self.node_and_type_dict[node] == "Fixed Support" :
                self.displacement_list.insert(node * 3, 0)
                self.displacement_list.insert(node * 3 + 1, 0)
                self.displacement_list.insert(node * 3 + 2, 0)
        self.revized_system_displacement_vector = np.array(self.displacement_list).reshape((len(self.coordx)) * 3, 1)
        print(self.revized_system_displacement_vector)
        # data_dis = pd.DataFrame(self.revized_system_displacement_vector)
        # data_dis.to_excel("Yerdeğiştirme Vektörü.xlsx")
    def element_displacement(self):
        """Calculate Element Displacement"""
        self.displacement_elements_dict = {}
        self.transformed_displacement_dict = {}
        self.transformed_force_dict = {}
        self.element_load_dict = {}
        load_matrix = np.zeros((6,1))
        self.add_load_vector_dict = {}
        for i in range(0, self.element_ID - 1):
            element_displa_vector = self.revized_system_displacement_vector[i * 3: i * 3 + 6, 0]
            self.displacement_elements_dict[i+1] = element_displa_vector
            element_displa_vector = np.array(element_displa_vector).reshape(6, 1)
            self.transformation_matrix(self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(i + 1)[4],
                                       self.element_ID_area_inertia_elasticity_length_cos_sin_dict.get(i + 1)[5])
            transformed_displacement_element = np.dot(self.transformationMtrx, element_displa_vector)
            self.transformed_displacement_dict[i + 1] = transformed_displacement_element
        for i in range(1, self.element_ID):
            self.transformed_force_dict[i] = np.dot(self.transformed_stiffness_dict[i], self.transformed_displacement_dict[i])
            if self.element_ID_area_inertia_elasticity_length_cos_sin_dict[i][5] > 0:
                load_matrix[:3, 0] = self.element_load_vector[3:, 0]
                load_matrix[3:, 0] = self.element_load_vector[:3, 0]
                self.add_load_vector_dict[i] = load_matrix
            else:
                load_matrix = self.element_load_vector
                self.add_load_vector_dict[i] = load_matrix
            if (i - 1) in self.distributed_load_and_elements.keys():
                if self.distributed_load_and_elements[i - 1] < 0:
                    self.element_load_dict[i] = np.dot(self.transformed_stiffness_dict[i], self.transformed_displacement_dict[i]) - self.add_load_vector_dict[i]
                else:
                    self.element_load_dict[i] = np.dot(self.transformed_stiffness_dict[i],
                                                       self.transformed_displacement_dict[i]) + self.add_load_vector_dict[i]
            else:
                self.element_load_dict[i] = np.dot(self.transformed_stiffness_dict[i], self.transformed_displacement_dict[i])
            print("forces:",self.element_load_dict)
    def draw_displacement(self):
        """Drawing Displacements"""
        plt.figure(figsize=(4, 3))
        plt.xlim([min(self.coordx) - 1, max(self.coordx) + 1])
        plt.scatter(self.coordx, self.coordy)
        for i in range(self.element_ID):
            plt.annotate(f"{self.displacement_list[i * 3]:.2E}\n, {self.displacement_list[i * 3 + 1]:.2E}\n, {self.displacement_list[i * 3 + 2]:.2E}", (self.coordx[i], self.coordy[i]), fontsize=7)
        plt.savefig("displacement.png", dpi=100)
        self.uiMenu.displacement_label.setPixmap(QtGui.QPixmap("displacement.png"))
    # def draw_axial_loads(self):
    #     """Drawing Axial Loads"""
    #     ...
        # for i in self.element_load_dict.keys():
        #     data_element_loads = pd.DataFrame(self.element_load_dict[i])
        #     data_element_loads.to_excel(f"{i}. Eleman İç Kuvvetleri.xlsx", sheet_name="Element Forces")
        # plt.figure(figsize=(4, 3))
        # for i in range(len(self.element_side_left)):
        #     plt.plot([self.node_coords[self.element_side_left[i]][0], self.node_coords[self.element_side_right[i]][0]],
        #              [self.node_coords[self.element_side_left[i]][1], self.node_coords[self.element_side_right[i]][1]], "black", linewidth=4.0)
        # # plt.plot(self.x_coor_elem, self.y_coor_elem, "black", linewidth=4.0)
        # plt.savefig("axial.png", bbox_inches='tight', dpi=100)
        # self.uiMenu.axial_label.setPixmap(QtGui.QPixmap("axial.png"))
        # for i in self.element_load_dict.keys():
        #     starting_axial_load = self.element_load_dict[i][0]
        #     last_axial_load = self.element_load_dict[i][3]
        #     if starting_axial_load > last_axial_load:
        #         increment_value = -0.01
        #     else:
        #         increment_value = 0.01
        #     load_increment = np.arange(starting_axial_load, last_axial_load, increment_value)
        #     plt.figure(figsize=(4, 3))
        #     plt.plot([self.node_coords[self.element_side_left[i]][0], self.node_coords[self.element_side_right[i]][0]],
        #              [self.node_coords[self.element_side_left[i]][1], self.node_coords[self.element_side_right[i]][1]], load_increment, "red")
        #     plt.show()
    # def draw_shear_forces(self):
    #     ...
    def create_report(self):
        """Create Report"""
        # """DÜĞÜM NOKTALARI"""
        with open("Rapor.txt", "w") as report:
            report.write("================================================================================================================================================\nDüğüm No:\tKoordinat X:\tKoordinat Y:\n================================================================================================================================================\n")
        with open("Rapor.txt", "a") as report:
            for i in range(len(self.node_coords)):
                report.write(f"{i + 1}.\t\t{self.node_coords[i][0]}\t\t{self.node_coords[i][1]}\n")
        # """ELEMAN KESİTLERİ"""
        with open("Rapor.txt", "a") as report:
            report.write("\n================================================================================================================================================\nEleman Adı\tAlanı:\t\tElastisite Modülü:\tAtalet Momenti\n================================================================================================================================================\n")
        with open("Rapor.txt", "a") as report:
            for i in self.element_prop.keys():
                report.write(f"{i}\t\t{self.element_prop[i][0]}\t\t\t{float(self.element_prop[i][2]):.2E}\t{float(self.element_prop[i][1]):.2E}\n")
        # """ELEMAN RİJİTLİK MATRİSLERİ"""
        with open("Rapor.txt", "a") as report:
            report.write("\n================================================================================================================================================\nEleman Rijitlik Matrisleri\n================================================================================================================================================\n")
        with open("Rapor.txt", "a") as report:
            for i in self.element_stiffness_matrix_dict.keys():
                report.write(f"\n{i}. Eleman\n{self.element_stiffness_matrix_dict[i]}\n")
        # """ELEMAN TRANSFORMASYON MATRİSLERİ"""
        with open("Rapor.txt", "a") as report:
            report.write("\n================================================================================================================================================\nEleman Transformasyon Matrisleri\n================================================================================================================================================\n")
        with open("Rapor.txt", "a") as report:
            for i in self.element_transformation_matrix_dict.keys():
                report.write(f"\n{i}. Eleman\n{self.element_transformation_matrix_dict[i]}\n")
        # """TOPLAM SİSTEM RİJİTLİK MATRİSİ"""
        with open("Rapor.txt", "a") as report:
            report.write(f"\n================================================================================================================================================\nToplam Sistem Rijitlik Matrisi\n================================================================================================================================================\n\n{self.system_stiffness_matrix}\n")
        # """REVİZE SİSTEM RİJİTLİK MATRİSİ"""
        with open("Rapor.txt", "a") as report:
            report.write(f"\n================================================================================================================================================\nRevize Sistem Rijitlik Matrisi\n================================================================================================================================================\n\n{self.revized_StiffnessMtrx}\n")
        # """REVİZE SİSTEM YÜK VEKTÖRÜ"""
        with open("Rapor.txt", "a") as report:
            report.write(f"\n================================================================================================================================================\nRevize Sistem Yük Vektörü\n================================================================================================================================================\n\n{self.revized_LoadMtrx}\n")
        # """YERDEĞİŞTİRME VEKTÖRÜ"""
        with open("Rapor.txt", "a") as report:
            report.write("\n================================================================================================================================================\nYerdeğiştirme Vektörü\n================================================================================================================================================\nDüğüm Noktası:\t\tUx :\t\tUy :\t\t\tR :\n")
        with open("Rapor.txt", "a") as report:
            for i in range(0, len(self.transformed_displacement_dict.keys()) + 1):
                report.write(f"{i + 1}\t\t{float(self.revized_system_displacement_vector[i*3]):.4E}\t\t{float(self.revized_system_displacement_vector[i*3 + 1]):.4E}\t\t{float(self.revized_system_displacement_vector[i*3 + 2]):.4E}\n")
        with open("Rapor.txt", "a") as report:
            report.write("\n================================================================================================================================================\nEleman İç Kuvvetleri\n================================================================================================================================================\nEleman No:\tUÇ :\t\tN :\t\tT :\t\tM :\t\t UÇ :\t\tN :\t\tT :\t\tM :\n")
        with open("Rapor.txt", "a") as report:
            for i in self.element_load_dict.keys():
                # if self.element_ID_area_inertia_elasticity_length_cos_sin_dict[i][5] == 0:
                #     report.write(f"{i}\t\tsol\t\t{self.element_load_dict[i][1]}\t\t{self.element_load_dict[i][0]}\t\t{self.element_load_dict[i][2]}\t\tsağ\t\t{self.element_load_dict[i][4]}\t\t{self.element_load_dict[i][3]}\t\t{self.element_load_dict[i][5]}\n")
                # else:
                report.write(f"{i}\t\tsol\t\t{float(self.element_load_dict[i][0]):.3E}\t{float(self.element_load_dict[i][1]):.3E}\t{float(self.element_load_dict[i][2]):.3E}\tsağ\t\t{float(self.element_load_dict[i][3]):.3E}\t{float(self.element_load_dict[i][4]):.3E}\t{float(self.element_load_dict[i][5]):.3E}\n")
    # def draw_moment_diagram(self):
    #     ...
        # """Drawing Moment Diagrams"""
        # plt.figure(figsize=(4, 3))
        # for i in range(len(self.element_side_left)):
        #     plt.plot([self.node_coords[self.element_side_left[i]][0], self.node_coords[self.element_side_right[i]][0]],
        #              [self.node_coords[self.element_side_left[i]][1], self.node_coords[self.element_side_right[i]][1]],
        #              "black", linewidth=4.0)
        # plt.savefig("moment.png", dpi=100)
        # self.uiMenu.moment_label.setPixmap(QtGui.QPixmap("moment.png"))
def app():
    """Define App for Initializing"""
    app_win = QtWidgets.QApplication(sys.argv)
    window = FiniteObject()
    window.show()
    sys.exit(app_win.exec_())
app()
