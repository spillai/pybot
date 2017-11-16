var urlParams;
var f1, f2;
var container, camera, controls, scene, renderer;
var ws;

var mouse = new THREE.Vector2();
var hoverCamera, raycaster, parentTransform;
// var capturer = null;
// var rec_button; 
var scene_group, grid_group;
var pointCloudMaterial, lineMaterial;

var obj_axes_geom = null;
var obj_collections = {};
var pc_collections = {};
var pc_obj_lut = {};

var collections_visibles = {};
var collections_visibles_lut = {};
var collections_material = {};

var savedOptions = {
    pointSize: 0,
    drawGrid: false,
    followCamera: false   
};

var options = {
    pointSize: 0.02,
    drawGrid: true,
    followCamera: true,
    animationSpeed: 1.0,
};


$('#loading').remove();
init();
animate();
// getData();

function addDatGui(){
    var gui = new dat.GUI();

    f1 = gui.addFolder('Options');

		// Point Size 
    f1.add(options, 'pointSize', 0, 0.25)
				.name('Point Size')
        .listen()
        .onChange(function(value) {
						pointCloudMaterial.size = value;
						render();
				});

		// Draw grid
    f1.add(options, 'drawGrid')
        .name('Show Grid')
        .listen()
				.onChange(function(value) {
						options.drawGrid = value;
						grid_group.visible = value;
						render();
				});

		// Follow camera
    f1.add(options, 'followCamera')
        .name('Follow Camera')
        .listen()
				.onChange(function(value) {
						options.followCamera = value;
						render();
				});				
    f1.open();

    f2 = gui.addFolder('Collections');
    f2.open();
    gui.close();
}

function convertTypedArray(src, type) {
    var buffer = new ArrayBuffer(src.byteLength);
    var baseView = new src.constructor(buffer).set(src);
    return new type(buffer);
}

function split_channel_data(ch_data) {
    val = ' '.charCodeAt(0);
    for (var i=0, L=ch_data.length; i < L; i++) {
        if (ch_data[i] == val) {
            return { channel: ch_data.slice(0,i),
                     data: ch_data.slice(i+1) }
        }
    }
    return {channel: -1, data: -1};
}

function update_camera_pose(msg) {
    if (!options.followCamera)
        return;
    
    // Place camera
    var mat = new THREE.Matrix4().makeRotationFromQuaternion(
        new THREE.Quaternion(msg.orientation[1],
                             msg.orientation[2],
                             msg.orientation[3],
                             msg.orientation[0])).transpose();

    var d = mat.elements;
    var ya = new THREE.Vector3(d[1], d[5], d[9]).negate();
    var za = new THREE.Vector3(d[2], d[6], d[10]);
    controls.goto_up(
        new THREE.Vector3(msg.pos[0], msg.pos[1], msg.pos[2]),
        new THREE.Vector3(za.x * 1 + msg.pos[0],
                                    za.y * 1 + msg.pos[1],
                          za.z * 1 + msg.pos[2]),
        new THREE.Vector3(ya.x, ya.y, ya.z)
    );
    
}

function add_points_to_scene_group(msg) {
    // Note: All points from the same channel are associated with the
    // same frame_id (i.e. collection_id=uuid(pose_channel), element_id=0,...n)
    
    // Clean up point clouds for corresponding channel (indexed by msg.id)
    if (msg.reset && msg.id in pc_obj_lut) {

        // Tuple (element_group, point_cloud)
        for (var key in pc_obj_lut[msg.id]) {
            gp_pc = pc_obj_lut[msg.id][key];
            gp_pc[0].remove(gp_pc[1]);
        }
        delete pc_obj_lut[msg.id];
    }

    // Initialize pc-obj LUT
    if (!(msg.id in pc_obj_lut)) {
        pc_obj_lut[msg.id] = [];
    }
    
    // Render points
    for (var i = 0; i < msg.pointLists.length; ++i) {
        var pc = msg.pointLists[i];

        // Find collection_id, and element_id pose
        try {
            cid = pc.collection, eid = pc.elementId;
            var element_group = obj_collections[cid][eid];
        } catch (err) {
            console.log('Error finding collection, and element_id ' +
                        cid + ':' + eid);
            return;
        }

        // Convert bytes to float32array
        var pointsf = convertTypedArray(pc.points, Float32Array);
        var colorsf = convertTypedArray(pc.colors, Float32Array);
        
        // Add points into buffer geometry
        var geom = new THREE.BufferGeometry();
        geom.addAttribute(
            'position',
            new THREE.BufferAttribute(pointsf, 3));
        geom.addAttribute(
            'color',
            new THREE.BufferAttribute(colorsf, 3));

        var item; 

        // Render points
        switch (msg.type) {
        case point3d_list_collection_t.getEnum('point_type').POINT:
            item = new THREE.Points(
                geom, pointCloudMaterial);
            break;
            
        // Render lines
        case point3d_list_collection_t.getEnum('point_type').LINES:
            item = new THREE.LineSegments(
                geom, lineMaterial, THREE.LinePieces);
            break;
            
        // Render triangles
        case point3d_list_collection_t.getEnum('point_type').TRIANGLES:
            // Create triangles and compute normals
            for (var j = 0, pc_sz = pc.points.length / 3;
                 j < pc_sz; ++j) {
                geom.faces.push(
                    new THREE.Face3( 3*j, 3*j+1, 3*j+2 ));
            }
            mesh_material = new THREE.MeshBasicMaterial({
                color: 0xFFFF00,
            });
            item = new THREE.Mesh(geom, mesh_material);
            break;

        default:
            console.log('Unknown type ' + msg.type);
        }

				// Add point cloud material
				
        // For every point cloud added, maintain the corresponding
        // element_group it belongs to (for future removal purposes)
        pc_obj_lut[msg.id].push([element_group, item]);
        element_group.add(item);
        
        // Element group culling
        element_group.frustumCulled = true;
        
        // Add group to scene
        scene_group.add(element_group);
				        
    }
}

function add_objects_to_scene_group(msg) {

    // Clean up poses for corresponding channel (indexed by msg.id)
    if (msg.reset && msg.id in obj_collections) {
        for (var obj_id in obj_collections[msg.id]) {
            // Reomve obj from scene_group
            scene_group.remove(obj_collections[msg.id][obj_id]);
        }
        delete obj_collections[msg.id];
    }

    // Retreive object collection
    // Object.keys(obj_collections_lut).length == 0
    if (!(msg.id in obj_collections)) {
        obj_collections[msg.id] = {};
    }
    
    // Render poses
    for (var i = 0; i < msg.objs.length; ++i) {
        var obj = msg.objs[i];

        // Create object group for obj_id
        var update = false;
        if (!(obj.id in obj_collections[msg.id])) { 
            obj_collections[msg.id][obj.id] = new THREE.Object3D();
            // console.log('adding element ' + msg.id + ':' + obj.id);
        } else {
            update = true;
            // console.log('updating element ' + msg.id + ':' + obj.id);
        }
        
        // Transform obj_id
        var obj_group = obj_collections[msg.id][obj.id];
        obj_group.setRotationFromEuler(
            new THREE.Euler(obj.roll, obj.pitch, obj.yaw, 'ZYX'));
        obj_group.position.copy(new THREE.Vector3(obj.x, obj.y, obj.z));

        // First time add
        if (!update) {
            // Add axes to obj_id
            obj_group.add(getAxes(0.2));

            // Add obj_id to scene
            scene_group.add(obj_group);
        }
    }
    scene_group.frustumCulled = true;
}

function addGridAxes() {
    // add the three markers to the axes
    addAxis(new THREE.Vector3(1, 0, 0));
    addAxis(new THREE.Vector3(0, 1, 0));
    addAxis(new THREE.Vector3(0, 0, 1));
}

function addAxis(axis) {
    // create the cylinders for the objects
    var shaftRadius = 0.02;
    var headRadius = 0.04;
    var headLength = 0.1;

    var lineGeom = new THREE.CylinderGeometry(
        shaftRadius, shaftRadius, 1);
    var headGeom = new THREE.CylinderGeometry(
        0, headRadius, headLength);

    // set the color of the axis
    var color = new THREE.Color();
    color.setRGB(axis.x, axis.y, axis.z);
    var material = new THREE.MeshBasicMaterial({
      color : color.getHex()
    });

    var axis_group = new THREE.Object3D();
    
    // setup the rotation information
    var rotAxis = new THREE.Vector3();
    rotAxis.crossVectors(axis, new THREE.Vector3(0, -1, 0));
    var rot = new THREE.Quaternion();
    rot.setFromAxisAngle(rotAxis, 0.5 * Math.PI);

    // create the arrow
    var arrow = new THREE.Mesh(headGeom, material);
    arrow.matrix.makeRotationFromQuaternion(rot);
    arrow.matrix.setPosition(axis.multiplyScalar(1).clone());
    arrow.matrixAutoUpdate = false;
    axis_group.add(arrow);

    // create the line
    var line = new THREE.Mesh(lineGeom, material);
    line.matrix.makeRotationFromQuaternion(rot);
    line.matrix.setPosition(axis.multiplyScalar(0.5).clone());
    line.matrixAutoUpdate = false;
    axis_group.add(line);

    // add axis to group
    grid_group.add(axis_group);
}

function getAxes(sz) {
    if (!obj_axes_geom) { 
        obj_axes_geom = new THREE.Geometry();
        obj_axes_geom.vertices = [
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(sz, 0, 0),
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, sz, 0),
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 0, sz)
        ];
        obj_axes_geom.colors = [
            new THREE.Color( 0xff0000 ),
            new THREE.Color( 0xff0000 ),
            new THREE.Color( 0x00ff00 ),
            new THREE.Color( 0x00ff00 ),
            new THREE.Color( 0x0000ff ),
            new THREE.Color( 0x0000ff )
        ];
    }

    // Return new axis with cached geometry
    var axis = new THREE.LineSegments(
        obj_axes_geom, lineMaterial, THREE.LinePieces);
    return axis;
}

function init() {

    // -------------------------------------------
    // Load vs proto
    protobuf.load("vs.proto", function(err, root) {
        if (err)
            throw err;
        
        // Obtain a message type
        message_t = root.lookupType("vs.message_t");
        pose_t = root.lookupType("vs.pose_t");
        obj_collection_t = root.lookupType("vs.obj_collection_t");
        point3d_list_collection_t = root.lookupType("vs.point3d_list_collection_t");
        
    });

    // -------------------------------------------
    // Connect to Web Socket
    ws = new WebSocket("ws://localhost:9001/");
    ws.binaryType = 'arraybuffer';
    
    ws.onmessage = function(e) {
        if (e.data instanceof ArrayBuffer) {
            // Create buffer
            buf = new Uint8Array(e.data);
            
            // Split channel, and data
            msg_buf = split_channel_data(buf);
            ch_str = String.fromCharCode.apply(null, msg_buf.channel);

						var msg_id = null;
						var msg_name = null;
						var msg_collection = null;
						
            // Decode based on channel 
            switch(ch_str) {
            case 'CAMERA_POSE':
                msg = pose_t.decode(msg_buf.data);
                update_camera_pose(msg);
                break;
                
            case 'POINTS_COLLECTION':
                msg = point3d_list_collection_t.decode(msg_buf.data);
                add_points_to_scene_group(msg);
								msg_id = msg.id;
								msg_name = msg.name;
								break;
                
            case 'OBJ_COLLECTION':
                msg = obj_collection_t.decode(msg_buf.data);
                add_objects_to_scene_group(msg);
								msg_id = msg.id;
								msg_name = msg.name;
                break;
                
            case 'RESET_COLLECTIONS':
                console.log('<' + ch_str + '>');

                // Clean up point clouds
                pc_obj_lut = {};
                obj_collections = {};

								// Clean up collections folder
								for (var key in collections_visibles_lut) {
										f2.remove(collections_visibles_lut[key]);
								}
								collections_visibles = {};
								collections_visibles_lut = {};
								
                // Recursively delete all objects in the scene graph
                scene_group.traverse(function(child){
                    if (child.geometry != undefined) {
                        child.material.dispose();
                        child.geometry.dispose();
                    }
                });

                // Remove scene group
                scene.remove(scene_group);
                addEmptyScene();
                
                break;
                
            default:
                console.log('Unknown channel / decoder ' + ch_str);
            }

						// Add checkbox for relevant collections
						if (msg_id != null && !(msg_id in collections_visibles_lut)) {
								collections_visibles[msg_id] = true;
								collections_visibles_lut[msg_id] = f2
										.add(collections_visibles, msg_id)
										.name(msg_name)
										.listen()
										.onChange(function(value) {

												switch(ch_str) {
												case 'OBJ_COLLECTION':
														for (var key in obj_collections[msg_id]) {
																obj_collections[msg_id][key].visible = value;
														}
														break;

												case 'POINTS_COLLECTION':
														for (var key in pc_obj_lut[msg_id]) {
																// tuple = (element_group, point_cloud)
																tuple = pc_obj_lut[msg_id][key];
																tuple[0].visible = value;
														}
														break;
														
												default:
														break
														
												}
												render();
												
										});

						}
						
            // Re-render scene
            render();
            
        }
    };
    
    ws.onclose = function() {
        // output("onclose");
    };
    
    ws.onerror = function(e) {
        // output("onerror");
        console.log(e)
    };

    // initialize renderer
    initRenderer();

    // TODO: Image viewer (see onionmaps reference)
    
}

function addEmptyScene() {
    // Create scene group
    scene_group = new THREE.Object3D();
    scene_group.name = 'collections_scene';
    scene.add(scene_group);
}

function initRenderer() {
    raycaster = new THREE.Raycaster();
    raycaster.precision = 0.01;

    // TODO: optional preserveDrawingBuffer: true
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor( 0x202020, 0.0);
    // renderer.sortObjects = false;

    container = document.getElementById( 'ThreeJS' );
    container.appendChild(renderer.domElement);

    camera = new THREE.PerspectiveCamera(
        70, window.innerWidth / window.innerHeight, 0.03, 10000);
    camera.position.x = 50;
    camera.position.y = 50;
    camera.position.z = 50;
    camera.far = 200; // Setting far frustum (for culling)
    camera.up = new THREE.Vector3(0,0,1);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.addEventListener('change', render);

    window
        .addEventListener(
            'resize', onWindowResize, false);
    renderer.domElement
        .addEventListener(
            'mousemove', onDocumentMouseMove, false);
    
    // Set materials
    pointCloudMaterial = new THREE.PointsMaterial({
        size: options.pointSize,
        vertexColors: true,
    });
    lineMaterial = new THREE.LineBasicMaterial({
        color: 0xffffff,
        opacity: 1,
        linewidth: 3,
        vertexColors: THREE.VertexColors
    });
    

    // Axis
    grid_group = new THREE.Object3D();
    addGridAxes(); // grid_group.add(getAxes(1));

    // Ground grid
    {
        var linegeo = new THREE.Geometry();
        var N = 50;
        var scale = 5;
        for (var i = 0; i <= 2 * N; ++i) {
            linegeo.vertices.push(
                new THREE.Vector3(scale * (i - N), scale * (-N), 0),
                new THREE.Vector3(scale * (i - N), scale * ( N), 0),
                new THREE.Vector3(scale * (-N), scale * (i - N), 0),
                new THREE.Vector3(scale * ( N), scale * (i - N), 0)
            );
        }
        var lmaterial = new THREE.LineBasicMaterial({color:
                                                        0x555555});
        var line = new THREE.LineSegments(
            linegeo, lmaterial,
            THREE.LinePieces);
        // line.receiveShadow = true;
        grid_group.add(line);
    }
    grid_group.name = 'grid';
    // grid_group.frustumCulled = true;

    scene = new THREE.Scene();
    scene.add(grid_group);

    // Create empty scene
    addEmptyScene();

    // Add controls
    addDatGui();

    render();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    render();
}

function onDocumentMouseMove(event) {
    event.preventDefault();
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = - (event.clientY / window.innerHeight) * 2 + 1;
    render();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
}

function render() {
    // Render.
    renderer.render(scene, camera);
}
