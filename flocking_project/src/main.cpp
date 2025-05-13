/**
* @file      main.cpp
* @brief     Example Boids flocking simulation for CIS 565
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
* @copyright University of Pennsylvania
*/

#include "main.hpp"

// For additional timing tests
#include <vector>
#include <iostream>
#include <chrono> // Added for non-GLFW timing

// ================
// Configuration
// ================

#define VISUALIZE 1

// Default values (can be overridden by command-line arguments)
int N_FOR_VIS = 25000;
int simulationMethod = 2; // 0 = Naive, 1 = Scattered Grid, 2 = Coherent Grid
const float DT = 0.1f;  // Reduced from 0.2f for smoother motion

// New flags for controlling execution mode
bool enableVisualization = true;
bool perfTestMode = false;
bool perfTestBlockSize = false;

// Standard boid sizes to test
std::vector<int> standardBoidSizes = {5000, 10000, 25000, 50000, 70000, 100000, 500000, 1000000, 2500000, 5000000};

// Print usage information for command-line arguments
void printUsage() {
  std::cout << "Usage: " << std::endl;
  std::cout << "  ./bin/cis565_boids [--mode <mode>] [--boids <count>]" << std::endl;
  std::cout << "      Run with visualization using specified mode and boid count" << std::endl;
  std::cout << "      <mode>: naive, scattered, or coherent (default: naive)" << std::endl;
  std::cout << "      <count>: Number of boids (default: 5000)" << std::endl;
  std::cout << std::endl;
  std::cout << "  ./bin/cis565_boids --perf-test <mode>" << std::endl;
  std::cout << "      Run performance tests for all boid sizes with specified method" << std::endl;
  std::cout << "      <mode>: naive, scattered, or coherent" << std::endl;
  std::cout << std::endl;
  std::cout << "  ./bin/cis565_boids --perf-test <mode> <boid_count>" << std::endl;
  std::cout << "      Run performance test for a single boid count" << std::endl;
  std::cout << std::endl;
  std::cout << "  ./bin/cis565_boids --perf-test-block-size" << std::endl;
  std::cout << "      Run block size performance test for all three methods with 25,000 boids" << std::endl;
  std::cout << std::endl;
  std::cout << "  Examples:" << std::endl;
  std::cout << "    ./bin/cis565_boids --mode coherent --boids 10000" << std::endl;
  std::cout << "    ./bin/cis565_boids --perf-test naive" << std::endl;
  std::cout << "    ./bin/cis565_boids --perf-test scattered 10000" << std::endl;
  std::cout << "    ./bin/cis565_boids --perf-test-block-size" << std::endl;
}

// Updated block size performance test to use chrono instead of GLFW for timing
void runBlockSizeTest(int numBoids, int numSteps) {
  std::vector<int> methods = {0, 1, 2};
  std::vector<std::string> methodNames = {"Naive", "Scattered Grid", "Coherent Grid"};
  std::vector<int> blockSizes = {32, 64, 128, 256, 512, 1024};
  
  std::cout << "\n===============================================================" << std::endl;
  std::cout << "BLOCK SIZE PERFORMANCE TEST WITH " << numBoids << " BOIDS" << std::endl;
  std::cout << "===============================================================\n" << std::endl;
  
  for (int method : methods) {
    std::cout << "\n===========================================" << std::endl;
    std::cout << "TESTING " << methodNames[method] << " IMPLEMENTATION" << std::endl;
    std::cout << "===========================================\n" << std::endl;
    
    for (int blockSize : blockSizes) {
      std::cout << "Block size: " << blockSize << std::endl;
      
      simulationMethod = method;
      Boids::currentBlockSize = blockSize;
      Boids::initSimulation(numBoids);
      
      // Warmup runs
      for (int i = 0; i < 10; ++i) {
        if (method == 0) Boids::stepSimulationNaive(DT);
        else if (method == 1) Boids::stepSimulationScatteredGrid(DT);
        else Boids::stepSimulationCoherentGrid(DT);
      }
      
      // Timing test using CUDA events
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      
      for (int i = 0; i < numSteps; ++i) {
        if (method == 0) Boids::stepSimulationNaive(DT);
        else if (method == 1) Boids::stepSimulationScatteredGrid(DT);
        else Boids::stepSimulationCoherentGrid(DT);
      }
      
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float timeMs;
      cudaEventElapsedTime(&timeMs, start, stop);
      
      float avgTimePerStep = timeMs / numSteps;
      float fps = 1000.0f / avgTimePerStep;
      std::cout << "  Time: " << avgTimePerStep << " ms" << std::endl;
      std::cout << "  FPS: " << fps << std::endl;
      std::cout << std::endl;
      
      Boids::endSimulation();
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }
    
    std::cout << "Completed testing for " << methodNames[method] << " implementation." << std::endl;
  }
  
  std::cout << "\n===============================================================" << std::endl;
  std::cout << "Block size performance test completed." << std::endl;
  std::cout << "===============================================================" << std::endl;
}

/**
* C main function.
*/
int main(int argc, char* argv[]) {
  projectName = "565 CUDA Intro: Boids";

  // Default settings
  enableVisualization = true;
  perfTestMode = false;
  simulationMethod = 0; // Default to Naive
  N_FOR_VIS = 5000;     // Default to 5000 boids
  bool runAllBoidSizes = false;

  // Parse command-line arguments
  int i = 1;
  while (i < argc) {
    std::string arg = argv[i++];

    if (arg == "--mode") {
      if (i >= argc) {
        std::cerr << "Error: Missing mode after --mode" << std::endl;
        printUsage();
        return 1;
      }
      
      std::string mode = argv[i++];
      if (mode == "naive") {
        simulationMethod = 0;
      } else if (mode == "scattered") {
        simulationMethod = 1;
      } else if (mode == "coherent") {
        simulationMethod = 2;
      } else {
        std::cerr << "Error: Invalid simulation mode. Use 'naive', 'scattered', or 'coherent'." << std::endl;
        printUsage();
        return 1;
      }
    }
    else if (arg == "--boids") {
      if (i >= argc) {
        std::cerr << "Error: Missing count after --boids" << std::endl;
        printUsage();
        return 1;
      }
      
      try {
        N_FOR_VIS = std::stoi(argv[i++]);
        if (N_FOR_VIS <= 0) {
          std::cerr << "Error: Boid count must be positive" << std::endl;
          printUsage();
          return 1;
        }
      } catch (const std::invalid_argument&) {
        std::cerr << "Error: Invalid boid count format" << std::endl;
        printUsage();
        return 1;
      } catch (const std::out_of_range&) {
        std::cerr << "Error: Boid count out of range" << std::endl;
        return 1;
      }
    }
    else if (arg == "--perf-test") {
      enableVisualization = false;
      perfTestMode = true;
      
      if (i < argc) {
        std::string mode = argv[i++];
        
        // Check if it's a mode specification
        if (mode == "naive") {
          simulationMethod = 0;
        } else if (mode == "scattered") {
          simulationMethod = 1;
        } else if (mode == "coherent") {
          simulationMethod = 2;
        } else {
          // Not a recognized mode, check if it's a number (boid count)
          try {
            // Step back and reparse as a boid count
            i--;
            N_FOR_VIS = std::stoi(argv[i++]);
            if (N_FOR_VIS <= 0) {
              std::cerr << "Error: Boid count must be positive" << std::endl;
              printUsage();
              return 1;
            }
          } catch (const std::invalid_argument&) {
            std::cerr << "Error: Invalid argument after --perf-test. Expected 'naive', 'scattered', 'coherent', or a boid count." << std::endl;
            printUsage();
            return 1;
          } catch (const std::out_of_range&) {
            std::cerr << "Error: Boid count out of range" << std::endl;
            return 1;
          }
        }
        
        // Check if there's also a boid count argument
        if (i < argc) {
          try {
            N_FOR_VIS = std::stoi(argv[i++]);
            if (N_FOR_VIS <= 0) {
              std::cerr << "Error: Boid count must be positive" << std::endl;
              printUsage();
              return 1;
            }
          } catch (const std::invalid_argument&) {
            // If not a valid number, it might be another option, so step back
            i--;
          } catch (const std::out_of_range&) {
            std::cerr << "Error: Boid count out of range" << std::endl;
            return 1;
          }
        } else {
          // If no specific boid count is provided after the mode, run all boid sizes
          runAllBoidSizes = true;
        }
      } else {
        // No arguments after --perf-test, use default and run all boid sizes
        runAllBoidSizes = true;
      }
    }
    else if (arg == "--perf-test-block-size") {
      perfTestBlockSize = true;
      enableVisualization = false;
    }
    else {
      std::cerr << "Error: Unknown argument: " << arg << std::endl;
      printUsage();
      return 1;
    }
  }
  
  // Handle special performance test mode
  if (perfTestBlockSize) {
    runBlockSizeTest(25000, 100);
    return 0;
  }

  // Get method name for display
  std::string methodName;
  if (simulationMethod == 0) methodName = "Naive";
  else if (simulationMethod == 1) methodName = "Scattered Grid";
  else methodName = "Coherent Grid";
  
  // Handle the case to run tests with all boid sizes
  if (runAllBoidSizes) {
    std::vector<int> boidSizes = standardBoidSizes;
    std::cout << "Running performance tests for method: " << methodName << " with all boid sizes\n" << std::endl;
    
    for (int size : boidSizes) {
      N_FOR_VIS = size;
      std::cout << "\n========== Testing " << size << " boids ==========" << std::endl;
      
      if (init(argc, argv)) {
        mainLoop();
        Boids::endSimulation();
      } else {
        std::cerr << "Initialization failed for " << size << std::endl;
      }
    }
    
    return 0;
  }
  
  // Regular single run (visualization or single performance test)  
  if (enableVisualization) {
    std::cout << "Running simulation with " << N_FOR_VIS << " boids, method: " 
              << methodName << " (with visualization)" << std::endl;
  }

  if (init(argc, argv)) {
    mainLoop();
    Boids::endSimulation();
    return 0;
  } else {
    return 1;
  }
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
  // Set window title to "Student Name: [SM 2.0] GPU Name"
  cudaDeviceProp deviceProp;
  int gpuDevice = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpuDevice > device_count) {
    std::cout
    << "Error: GPU device number is greater than the number of devices!"
    << " Perhaps a CUDA-capable GPU is not installed?"
    << std::endl;
    return false;
  }
  cudaGetDeviceProperties(&deviceProp, gpuDevice);
  int major = deviceProp.major;
  int minor = deviceProp.minor;

  std::ostringstream ss;
  ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
  deviceName = ss.str();

  // Only initialize GLFW and OpenGL if visualization is needed
  if (enableVisualization) {
    // Window setup stuff
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
      std::cout
      << "Error: Could not initialize GLFW!"
      << " Perhaps OpenGL 3.3 isn't available?"
      << std::endl;
      return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
    if (!window) {
      glfwTerminate();
      return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
      return false;
    }

    // Initialize drawing state
    initVAO();

    // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
    // change the device ID.
    cudaGLSetGLDevice(0);

    cudaGLRegisterBufferObject(boidVBO_positions);
    cudaGLRegisterBufferObject(boidVBO_velocities);

    updateCamera();

    initShaders(program);

    glEnable(GL_DEPTH_TEST);
  }
  // Skip all GLFW initialization in performance mode
  // No need for GLFW timing in performance mode, we'll use std::chrono instead

  // Always initialize N-body simulation, regardless of visualization mode
  Boids::initSimulation(N_FOR_VIS);

  return true;
}

void initVAO() {

  std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (N_FOR_VIS)] };
  std::unique_ptr<GLuint[]> bindices{ new GLuint[N_FOR_VIS] };

  glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
  glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

  for (int i = 0; i < N_FOR_VIS; i++) {
    bodies[4 * i + 0] = 0.0f;
    bodies[4 * i + 1] = 0.0f;
    bodies[4 * i + 2] = 0.0f;
    bodies[4 * i + 3] = 1.0f;
    bindices[i] = i;
  }


  glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
  glGenBuffers(1, &boidVBO_positions);
  glGenBuffers(1, &boidVBO_velocities);
  glGenBuffers(1, &boidIBO);

  glBindVertexArray(boidVAO);

  // Bind the positions array to the boidVAO by way of the boidVBO_positions
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
  glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

  glEnableVertexAttribArray(positionLocation);
  glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  // Bind the velocities array to the boidVAO by way of the boidVBO_velocities
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
  glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(velocitiesLocation);
  glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

  glBindVertexArray(0);
}

void initShaders(GLuint * program) {
  GLint location;

  program[PROG_BOID] = glslUtility::createProgram(
    "shaders/boid.vert.glsl",
    "shaders/boid.geom.glsl",
    "shaders/boid.frag.glsl", attributeLocations, 2);
    glUseProgram(program[PROG_BOID]);

    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
      glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
      glUniform3fv(location, 1, &cameraPosition[0]);
    }
  }

  //====================================
  // Main loop
  //====================================
  void runCUDA() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
    // use this buffer

    if (enableVisualization) {
      float4 *dptr = NULL;
      float *dptrVertPositions = NULL;
      float *dptrVertVelocities = NULL;

      cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
      cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);

      // execute the kernel
      if (simulationMethod == 2) {
        Boids::stepSimulationCoherentGrid(DT);
      } else if (simulationMethod == 1) {
        Boids::stepSimulationScatteredGrid(DT);
      } else {
        Boids::stepSimulationNaive(DT);
      }

      // Copy positions and velocities to VBO for visualization
      Boids::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
      
      // unmap buffer object
      cudaGLUnmapBufferObject(boidVBO_positions);
      cudaGLUnmapBufferObject(boidVBO_velocities);
    } else {
      // In performance test mode, just run the simulation without OpenGL
      if (simulationMethod == 2) {
        Boids::stepSimulationCoherentGrid(DT);
      } else if (simulationMethod == 1) {
        Boids::stepSimulationScatteredGrid(DT);
      } else {
        Boids::stepSimulationNaive(DT);
      }
    }
  }

  void mainLoop() {
    double fps = 0;
    double timebase = 0;
    int frame = 0;
    
    // Performance metrics
    const int numFramesToMeasure = 100;
    double totalSimTime = 0.0;
    int failedLaunches = 0;

    // Only run unit test in visualization mode
    if (enableVisualization) {
      Boids::unitTest(); // Run the unit test only in visualization mode
    }
    
    if (perfTestMode) {
      // Performance test mode - no visualization, using chrono for timing
      try {
        for (int i = 0; i < numFramesToMeasure; i++) {
          // Use std::chrono for timing instead of glfwGetTime
          auto startTime = std::chrono::high_resolution_clock::now();
          
          // Execute simulation step directly
          runCUDA();
          
          cudaDeviceSynchronize();
          
          auto endTime = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> elapsed = endTime - startTime;
          totalSimTime += elapsed.count();
        }
      } catch (const std::exception& e) {
        std::cerr << "Error during simulation: " << e.what() << std::endl;
        failedLaunches++;
      } catch (...) {
        std::cerr << "Unknown error during simulation!" << std::endl;
        failedLaunches++;
      }
      
      // Calculate and output performance metrics
      double avgSimTime = totalSimTime / (numFramesToMeasure - failedLaunches);
      double avgFPS = (numFramesToMeasure - failedLaunches) / totalSimTime;
      
      std::string methodName;
      if (simulationMethod == 0) methodName = "Naive";
      else if (simulationMethod == 1) methodName = "Scattered Grid";
      else methodName = "Coherent Grid";
      
      std::cout << "\n=========== PERFORMANCE METRICS ===========" << std::endl;
      std::cout << "Boid Count: " << N_FOR_VIS << std::endl;
      std::cout << "Method: " << methodName << std::endl;
      std::cout << "Average FPS: " << avgFPS << std::endl;
      std::cout << "Average Simulation Time per Step (ms): " << (avgSimTime * 1000.0) << std::endl;
      std::cout << "Failed Launches: " << failedLaunches << std::endl;
      std::cout << "=========================================" << std::endl;
    } else {
      // Interactive visualization mode - using GLFW for timing and rendering
      while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        frame++;
        double time = glfwGetTime();

        if (time - timebase > 1.0) {
          fps = frame / (time - timebase);
          timebase = time;
          frame = 0;
        }

        runCUDA();

        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps] " << deviceName;
        glfwSetWindowTitle(window, ss.str().c_str());

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(program[PROG_BOID]);
        glBindVertexArray(boidVAO);
        glPointSize((GLfloat)pointSize);
        glDrawElements(GL_POINTS, N_FOR_VIS + 1, GL_UNSIGNED_INT, 0);
        glPointSize(1.0f);

        glUseProgram(0);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
      }
      
      // Clean up GLFW
      glfwDestroyWindow(window);
      glfwTerminate();
    }
  }

  void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
  }

  void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GL_TRUE);
    }
  }

  void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  }

  void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (leftMousePressed) {
      // compute new camera parameters
      phi += (xpos - lastX) / width;
      theta -= (ypos - lastY) / height;
      theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
      updateCamera();
    }
    else if (rightMousePressed) {
      zoom += (ypos - lastY) / height;
      zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
      updateCamera();
    }

	lastX = xpos;
	lastY = ypos;
  }

  void updateCamera() {
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.z = zoom * cos(theta);
    cameraPosition.y = zoom * cos(phi) * sin(theta);
    cameraPosition += lookAt;

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
    projection = projection * view;

    GLint location;

    glUseProgram(program[PROG_BOID]);
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
      glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
  }