class Network ( object ) :
    def __init__ ( self , num_nodes ) :
        self . adj = { i :set () for i in range ( num_nodes ) }
    
    def add_edge ( self , i , j ) :
        self . adj [ i ]. add ( j )
        self . adj [ j ]. add ( i )

    def neighbors ( self , i ) :
            
        return self . adj [ i ]


    def edge_list ( self ) :
        return [( i , j ) for i in self . adj for j in self . adj [ i ] if i < j ]
    
net = Network(3)
net.add_edge(0, 1)
net.add_edge(1, 2)

print(net.neighbors(1))      # Output: {0, 2}
print(net.edge_list())  