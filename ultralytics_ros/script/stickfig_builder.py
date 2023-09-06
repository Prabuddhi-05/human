#!/usr/bin/env python3

# For constructing stick figures based on world coordinates of body parts 

import math
import numpy as np

class stickfig_builder:
    def __init__(self, person_world_coordinates):
        # Takes the initial world coordinates of body parts 
        self.A = person_world_coordinates["A"] # Nose
        self.B = person_world_coordinates["B"] # Mid-shoulder
        self.C = person_world_coordinates["C"] # Left arm
        self.D = person_world_coordinates["D"] # Right arm
        self.E = person_world_coordinates["E"] # Mid-hip
        self.F = person_world_coordinates["F"] # Left ankle 
        self.G = person_world_coordinates["G"] # Right ankle
        self.stickfig = [self.A,self.B,self.C,self.D,self.E,self.F,self.G]
        
        # This list keeps track of the body parts that have been successfully added to the stick figure during its construction process.
	# Initially, it is empty as no body parts have been added yet. As the construction progresses, body parts are appended to this list.
	# This helps in identifying which body parts are still missing in the "baby" version of the stick figure.

        self.baby = [] 
        
        # This list defines the complete set of body parts that an "adult" version of the stick figure should have.
	# Each element corresponds to a specific body part, represented by a character label.
	# This list is used as a reference to determine whether the stick figure construction is complete or not.

        self.adult = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  
        self.adult_joints = [['A', 'B'],['B', 'C'],['B', 'D'],['B', 'E'],['E', 'F'],['E', 'G']]  # Links betweeen specific body joints
        
        ascii_value = ord('A') # Assigns the ASCII value of 'A' 
        for i, node in enumerate(self.stickfig): # Iterates through elements in 'self.stickfig' with their index
            if node != [0.0, 0.0, 0.0]: # Indicates the presence of a body part
                self.baby.append(chr(ascii_value+i)) # Appends a character label to 'self.baby' based on the ASCII value and index
        self.baby = sorted(self.baby)
                
    # Adds missing body parts to the "baby" stick figure to create a complete stick figure 
    def feed_baby(self):
        if 'B' in self.baby:
            if 'E' in self.baby:
                if 'A' not in self.baby:
                    # Adds A based on B (Adds AB length to point B)
                    #self.A = [self.B[0], self.B[1], self.B[2] + AB length]
                    self.A = [self.B[0], self.B[1], self.B[2] + 0.19275182485580444] 
                    self.baby.append('A')
                    return 'A'
                    
                # Finding C    
                if 'C' not in self.baby: 

                    if 'D' in self.baby:
                    #Extends BE and 'D' is the mirror of point C
                        # Calculates the direction vector of the line formed by B and E
                        direction_vector = np.array(self.E) - np.array(self.B)
                        direction_vector = direction_vector / np.linalg.norm(direction_vector)

                        # Calculates the midpoint of the line segment
                        midpoint = (np.array(self.B) + np.array(self.E)) / 2

                        # Calculates the vector between D and the midpoint
                        vector_to_midpoint = midpoint - np.array(self.D)

                        # Mirrors the vector around the direction vector
                        mirrored_vector = vector_to_midpoint - 2 * (np.dot(vector_to_midpoint, direction_vector)) * direction_vector

                        # Calculates the new C coordinates
                        self.C = midpoint + mirrored_vector
                        self.baby.append('C')
                        return 'C'
                        
                    if 'F' in self.baby:
                        # Reduces a length equal to 'BE' length from z of F 
                        self.C = [self.F[0], self.F[1], self.F[2] + 0.5606778264045715]
                        self.baby.append('C')
                        return('C')
                            
                    if 'G' in self.baby:
                        # Extends BE and 'G' is the mirror of point F
                        # Calculates the direction vector of the line formed by B and E
                        direction_vector = np.array(self.E) - np.array(self.B)
                        direction_vector = direction_vector / np.linalg.norm(direction_vector)

                        # Calculates the midpoint of the line segment
                        midpoint = (np.array(self.B) + np.array(self.E)) / 2

                        # Calculates the vector between G and the midpoint
                        vector_to_midpoint = midpoint - np.array(self.G)

                        # Mirrors the vector around the direction vector
                        mirrored_vector = vector_to_midpoint - 2 * (np.dot(vector_to_midpoint, direction_vector)) * direction_vector

                        # Calculates the new F coordinates 
                        self.F = midpoint + mirrored_vector
                        self.baby.append('F')
                        return 'F'
                                
                    if 'B' in self.baby:
                        # Adds an arbitrary 'BC cos 45' distance to y of B and reduces from z of B 
                        d = 0.88329790018526 * math.cos(math.radians(45))
                        self.C =  [self.B[0], self.B[1] + d , self.B[2] - d]  
                        self.baby.append('C')
                        return 'C' 
                                    
                # Finding D    
                if 'D' not in self.baby: 

                    if 'C' in self.baby:
                    #Extends BE and 'C' is the mirror of point D
                        # Calculates the direction vector of the line formed by B and E
                        direction_vector = np.array(self.E) - np.array(self.B)
                        direction_vector = direction_vector / np.linalg.norm(direction_vector)

                        # Calculates the midpoint of the line segment
                        midpoint = (np.array(self.B) + np.array(self.E)) / 2

                        # Calculates the vector between C and the midpoint
                        vector_to_midpoint = midpoint - np.array(self.C)

                        # Mirrors the vector around the direction vector
                        mirrored_vector = vector_to_midpoint - 2 * (np.dot(vector_to_midpoint, direction_vector)) * direction_vector

                        # Calculates the new D coordinates
                        self.D = midpoint + mirrored_vector
                        self.baby.append('D')
                        return 'D'
                        
                    if 'G' in self.baby:
                        # Reduces a length equal to 'BE' length from z of G 
                        self.D = [self.G[0], self.G[1], self.G[2] + 0.5606778264045715]
                        self.baby.append('D')
                        return('D')
                            
                    if 'F' in self.baby:
                        # Extends BE and 'F' is the mirror of point G
                        # Calculates the direction vector of the line formed by B and E
                        direction_vector = np.array(self.E) - np.array(self.B)
                        direction_vector = direction_vector / np.linalg.norm(direction_vector)

                        # Calculates the midpoint of the line segment
                        midpoint = (np.array(self.B) + np.array(self.E)) / 2

                        # Calculates the vector between F and the midpoint
                        vector_to_midpoint = midpoint - np.array(self.F)

                        # Mirrors the vector around the direction vector
                        mirrored_vector = vector_to_midpoint - 2 * (np.dot(vector_to_midpoint, direction_vector)) * direction_vector

                        # Calculates the new G coordinates 
                        self.G = midpoint + mirrored_vector
                        self.baby.append('G')
                        return 'G'
                        
                    if 'B' in self.baby:
                        # Adds an arbitrary 'BD cos 45' distance to y of B and reduces from z of B 
                        d = 0.795337842684871 * math.cos(math.radians(45))
                        self.D =  [self.B[0], self.B[1] + d , self.B[2] - d]  
                        self.baby.append('D')
                        return 'D' 
                         
                # Finding F    
                if 'F' not in self.baby: 

                    if 'G' in self.baby:
                    #Extends BE and 'G' is the mirror of point F
                        # Calculates the direction vector of the line formed by B and E
                        direction_vector = np.array(self.E) - np.array(self.B)
                        direction_vector = direction_vector / np.linalg.norm(direction_vector)

                        # Calculates the midpoint of the line segment
                        midpoint = (np.array(self.B) + np.array(self.E)) / 2

                        # Calculates the vector between G and the midpoint
                        vector_to_midpoint = midpoint - np.array(self.G)

                        # Mirrors the vector around the direction vector
                        mirrored_vector = vector_to_midpoint - 2 * (np.dot(vector_to_midpoint, direction_vector)) * direction_vector

                        # Calculates the new F coordinates
                        self.F = midpoint + mirrored_vector
                        self.baby.append('F')
                        return 'F'
                        
                    if 'C' in self.baby:
                        # Adds a length equal to 'BE' length from z of C 
                        self.F = [self.C[0], self.C[1], self.C[2] - 0.5606778264045715]
                        self.baby.append('F')
                        return('F')
                            
                    if 'D' in self.baby:
                        # Extends BE and 'D' is the mirror of point C
                        # Calculates the direction vector of the line formed by B and E
                        direction_vector = np.array(self.E) - np.array(self.B)
                        direction_vector = direction_vector / np.linalg.norm(direction_vector)

                        # Calculates the midpoint of the line segment
                        midpoint = (np.array(self.B) + np.array(self.E)) / 2

                        # Calculates the vector between D and the midpoint
                        vector_to_midpoint = midpoint - np.array(self.D)

                        # Mirrors the vector around the direction vector
                        mirrored_vector = vector_to_midpoint - 2 * (np.dot(vector_to_midpoint, direction_vector)) * direction_vector

                        # Calculates the new C coordinates 
                        self.C = midpoint + mirrored_vector
                        self.baby.append('C')
                        return 'C'
                        
                    if 'E' in self.baby:
                        # Adds an arbitrary 'EF cos 45' distance to y of E and reduces from z of E
                        d = 1.1241761800919687 * math.cos(math.radians(45))
                        self.F =  [self.E[0], self.E[1] + d , self.E[2] - d]  
                        self.baby.append('F')
                        return 'F' 
                         
                # Finding G    
                if 'G' not in self.baby: 

                    if 'F' in self.baby:
                    #Extends BE and 'F' is the mirror of point G
                        # Calculates the direction vector of the line formed by B and E
                        direction_vector = np.array(self.E) - np.array(self.B)
                        direction_vector = direction_vector / np.linalg.norm(direction_vector)

                        # Calculates the midpoint of the line segment
                        midpoint = (np.array(self.B) + np.array(self.E)) / 2

                        # Calculates the vector between F and the midpoint
                        vector_to_midpoint = midpoint - np.array(self.F)

                        # Mirrors the vector around the direction vector
                        mirrored_vector = vector_to_midpoint - 2 * (np.dot(vector_to_midpoint, direction_vector)) * direction_vector

                        # Calculates the new G coordinates
                        self.G = midpoint + mirrored_vector
                        self.baby.append('G')
                        return 'G'
                        
                    if 'D' in self.baby:
                        # Adds a length equal to 'BE' length from z of D 
                        self.G = [self.D[0], self.D[1], self.D[2] - 0.5606778264045715]
                        self.baby.append('G')
                        return('G')
                            
                    if 'C' in self.baby:
                        # Extends BE and 'C' is the mirror of point D
                        # Calculates the direction vector of the line formed by B and E
                        direction_vector = np.array(self.E) - np.array(self.B)
                        direction_vector = direction_vector / np.linalg.norm(direction_vector)

                        # Calculates the midpoint of the line segment
                        midpoint = (np.array(self.B) + np.array(self.E)) / 2

                        # Calculates the vector between C and the midpoint
                        vector_to_midpoint = midpoint - np.array(self.C)

                        # Mirrors the vector around the direction vector
                        mirrored_vector = vector_to_midpoint - 2 * (np.dot(vector_to_midpoint, direction_vector)) * direction_vector

                        # Calculates the new D coordinates 
                        self.D = midpoint + mirrored_vector
                        self.baby.append('D')
                        return 'D'
                         
                    if 'E' in self.baby:
                        # Adds an arbitrary 'EG cos 45' distance to y of E and reduces from y of G
                        d = 1.0979900381614365 * math.cos(math.radians(45))
                        self.G =  [self.E[0], self.E[1] + d , self.E[2] - d]  
                        self.baby.append('G') 
                        return 'G'
                                       
            else:
                # Adds E based on B (Subtract BE length to B) 
                #self.E = [self.B[0], self.B[1], self.B[2] - BE length]
                self.E = [self.B[0], self.B[1], self.B[2] - 0.5606778264045715]
                self.baby.append('E')
                return 'E'
        
        elif 'E' in self.baby:
            # Adds B by based on E
            #self.B = [self.E[0], self.E[1], self.E[2] + BE length]
            self.B = [self.E[0], self.E[1], self.E[2] + 0.5606778264045715]
            self.baby.append('B')
            return 'B'
            
        elif 'A' in self.baby:
            # Adds B by based on A
            #self.B = [self.A[0], self.A[1], self.A[2] - AB length]
            self.B = [self.A[0], self.A[1], self.A[2] - 0.19275182485580444]
            self.baby.append('B')
            return 'B'
        
        elif 'C' in self.baby and 'D' in self.baby:
            # Adds B by getting midpoint of CD and then adding to z-coordinate of midpoint of CD
            mid_arms = [6.484999895095825, -3.1555252075195312, 0.2537921741604805] 
            # The Euclidean distance between B and C
            H = 0.88329790018526
            # Calculates the Euclidean distance between C and D
            D = np.linalg.norm(np.array(self.C) - np.array(self.D))
            # Calculates the Euclidean distance between B and mid_arms
            L = np.sqrt(H**2 - (D/2)**2) 
            #self.B = Midpoint's z + L   
            self.B = [mid_arms[0], mid_arms[1], mid_arms[2] + L]   
            self.baby.append('B')
            return 'B'
            
        elif 'F' in self.baby and 'G' in self.baby:
            # Adds E by getting midpoint of FG and adding z-coordinate of midpoint of FG
            mid_ankles = [6.484999895095825, -3.1555252075195312, -0.6401627063751221] 
            # The Euclidean distance between E and F
            H = 1.1241761800919687
            # Calculate the Euclidean distance between F and G
            D = np.linalg.norm(np.array(self.F) - np.array(self.G))
            # Calculate the Euclidean distance between B and mid_arms
            L = np.sqrt(H**2 - (D/2)**2) 
            #self.E = Midpoint's z + L   
            self.E = [mid_ankles[0], mid_ankles[1], mid_ankles[2] + L]
            self.baby.append('E')
            return 'E'
        else:
            print("Skeleton Died! :(")
            return 'X'
    
    def raise_baby(self):  # Adds missing links to the stick figure 'baby' until it becomes a complete stick figure 'adult'
        count = 0
        status = 2
        
        self.new_coordinates = {
        'A': self.A,
        'B': self.B,
        'C': self.C,
        'D': self.D,
        'E': self.E,
        'F': self.F,
        'G': self.G
        }
        
        if self.baby == self.adult:# Check for cases 3 and 4
            status = self.assert_adult()
            if status != 4:
                status = 3
        while self.baby != self.adult and count < 7: # Continues while the "baby" stick figure is not yet complete and the count is below 7
            feed_status = self.feed_baby() # Adds missing body parts to the "baby" stick figure
            if feed_status == 'X':
                status = 1
            self.baby = sorted(self.baby)  # Sorts the elements in 'self.baby' alphabetically 
            count += 1
            print(self.baby)
            
        print("Congratulations! You have a complete skeleton!")

        self.new_coordinates = {
        'A': self.A,
        'B': self.B,
        'C': self.C,
        'D': self.D,
        'E': self.E,
        'F': self.F,
        'G': self.G
        }
        
        self.assert_adult()
        #print("New World Coordinates:")
        #for key, value in self.new_coordinates.items():
         #   print(f"{key}: [{value[0]}, {value[1]}, {value[2]}]")
        print("------------------------------")
        return self.new_coordinates, status

   # def assert_adult2(self):  # Verifies if the 'adult' stick figure is sensible
        #for joint in self.adult_joints:          
            #D = np.linalg.norm(np.array(self.new_coordinates[joint[1]]) - np.array(self.new_coordinates[joint[0]])) # The distances between body joints 
            #print(D)
            #if D > 1.5: # If the distances between body joints are more than 3.0 units, resets all the coordinates to zeros. 
                #self.new_coordinates = {
                #'A': [0.0, 0.0, 0.0],
                #'B': [0.0, 0.0, 0.0],
                #'C': [0.0, 0.0, 0.0],
                #'D': [0.0, 0.0, 0.0],
                #'E': [0.0, 0.0, 0.0],
                #'F': [0.0, 0.0, 0.0],
                #'G': [0.0, 0.0, 0.0]
                #}
                #return self.new_coordinates
                
    # Calculates a point at a distance L along the line defined by two input points p1 and p2           
    def generate_point_along_line(self, p1, p2, L):  
        # Calculates the vector between p1 and p2
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        delta_z = p2[2] - p1[2]
    
        # Calculates the distance between p1 and p2
        distance_p1_p2 = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    
        # Calculate the unit vector along the p1p2 line
        unit_vector_x = delta_x / distance_p1_p2
        unit_vector_y = delta_y / distance_p1_p2
        unit_vector_z = delta_z / distance_p1_p2
    
        # Calculates the new point coordinates
        new_point_x = p1[0] + unit_vector_x * L
        new_point_y = p1[1] + unit_vector_y * L
        new_point_z = p1[2] + unit_vector_z * L
    
        return new_point_x, new_point_y, new_point_z
    
    def assert_adult(self):  # Verifies if the created 'adult' stick figure is sensible by checking distances between body parts
    
        AB = np.linalg.norm(np.array(self.new_coordinates['A']) - np.array(self.new_coordinates['B'])) # Calculates the Euclidean distance between points A and B  
        BE = np.linalg.norm(np.array(self.new_coordinates['B']) - np.array(self.new_coordinates['E'])) # Calculates the Euclidean distance between points B and E
        BC = np.linalg.norm(np.array(self.new_coordinates['B']) - np.array(self.new_coordinates['C'])) # Calculates the Euclidean distance between points B and C
        BD = np.linalg.norm(np.array(self.new_coordinates['B']) - np.array(self.new_coordinates['D'])) # Calculates the Euclidean distance between points B and D  
        EF = np.linalg.norm(np.array(self.new_coordinates['E']) - np.array(self.new_coordinates['F'])) # Calculates the Euclidean distance between points E and F
        EG = np.linalg.norm(np.array(self.new_coordinates['E']) - np.array(self.new_coordinates['G'])) # Calculates the Euclidean distance between points E and G
        
        
        # Checks and adjusts distances between joints to ensure the stick figure's proportions are reasonable
        if BE > 0.7: # If the distance between points B and E is more than 0.7 
            # Reset the distance between B and E to a standard distance of 0.56 (The distance between B and E of the perfect skeleton)
            new_point_x, new_point_y, new_point_z = self.generate_point_along_line(self.new_coordinates['E'], self.new_coordinates['B'], 0.56) 
            self.new_coordinates['B'] = [new_point_x, new_point_y, new_point_z] # New coordinate of joint 'B'
            
        if AB > 0.3: 
            new_point_x, new_point_y, new_point_z = self.generate_point_along_line(self.new_coordinates['B'], self.new_coordinates['A'], 0.19)
            self.new_coordinates['A'] = [new_point_x, new_point_y, new_point_z]

        if BC > 1.0:  
            new_point_x, new_point_y, new_point_z = self.generate_point_along_line(self.new_coordinates['B'], self.new_coordinates['C'], 0.85)
            self.new_coordinates['C'] = [new_point_x, new_point_y, new_point_z]
            
        if BD > 1.0:  
            new_point_x, new_point_y, new_point_z = self.generate_point_along_line(self.new_coordinates['B'], self.new_coordinates['D'], 0.85)
            self.new_coordinates['D'] = [new_point_x, new_point_y, new_point_z]
            
        if EF > 1.2: 
            new_point_x, new_point_y, new_point_z = self.generate_point_along_line(self.new_coordinates['E'], self.new_coordinates['F'], 1.1)
            self.new_coordinates['F'] = [new_point_x, new_point_y, new_point_z]
            
        if EG > 1.2: 
            new_point_x, new_point_y, new_point_z = self.generate_point_along_line(self.new_coordinates['E'], self.new_coordinates['G'], 1.1)
            self.new_coordinates['G'] = [new_point_x, new_point_y, new_point_z]
        else:
            return 4
            
        return self.new_coordinates
                
if __name__ == "__main__":
    person_coordinates = {
        "nose": [1.0, 0.4, 0.2],# A
        "mid_shoulder": [0.9, 0.3, 0.1],#B
        "left_arm": [1.1, 0.2, 0.3],#C
        "right_arm": [0.8, 0.5, 0.1],#D
        "mid_hips": [0.0, 0.0, 0.000],#E
        "left_ankle": [0.6, 0.7, 0.3],#F
        "right_ankle": [0.0, 0.0, 0.0]#G
    }

    person1 = stickfig_builder(person_coordinates)
