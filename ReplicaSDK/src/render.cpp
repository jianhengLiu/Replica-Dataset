// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <EGL.h>
#include <PTexLib.h>
#include <filesystem>
#include <iostream>
#include <pangolin/image/image_convert.h>

#include "GLCheck.h"
#include "MirrorRenderer.h"

int main(int argc, char *argv[]) {
  ASSERT(argc == 3 || argc == 4 || argc == 5,
         "Usage: ./ReplicaRenderer mesh.ply /path/to/atlases [mirrorFile] "
         "[render_path]");

  const std::string meshFile(argv[1]);
  const std::string atlasFolder(argv[2]);
  ASSERT(pangolin::FileExists(meshFile));
  ASSERT(pangolin::FileExists(atlasFolder));

  std::string surfaceFile;
  if (argc == 4) {
    surfaceFile = std::string(argv[3]);
    ASSERT(pangolin::FileExists(surfaceFile));
  }

  std::string renderPath;
  std::vector<pangolin::OpenGlMatrix> renderPoses;
  std::filesystem::path file_path;
  if (argc == 5) {
    renderPath = std::string(argv[4]);
    ASSERT(pangolin::FileExists(renderPath));
    file_path = std::filesystem::path(renderPath).replace_extension();

    std::ifstream file(renderPath);
    std::string line;
    // x y z qx qy qz qw
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      float x, y, z, qx, qy, qz, qw;
      if (!(iss >> x >> y >> z >> qx >> qy >> qz >> qw)) {
        break;
      }
      Eigen::Quaterniond q(qw, qx, qy, qz);

      static Eigen::Matrix3d R_OpenCV_To_OpenGL = Eigen::Matrix3d::Identity();
      R_OpenCV_To_OpenGL << 1, 0, 0, 0, -1, 0, 0, 0, -1;

      static Eigen::Matrix3d R_2 = Eigen::Matrix3d::Identity();
      R_2 << 0, 1, 0, -1, 0, 0, 0, 0, 1;
      Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
      auto rot = q.toRotationMatrix();
      // T.topLeftCorner(3, 3) = rot * R_OpenCV_To_OpenGL * R_2;
      T.topLeftCorner(3, 3) = rot;
      T.topRightCorner(3, 1) = Eigen::Vector3d(x, y, z);
      pangolin::OpenGlMatrix pose(T);

      renderPoses.push_back(pose);
    }
  }

  // const int width = 1280;
  // const int height = 960;
  const int width = 1200;
  const int height = 680;
  bool renderDepth = true;
  float depthScale = 65535.0f * 0.1f;

  // Setup EGL
  EGLCtx egl;

  egl.PrintInformation();

  if (!checkGLVersion()) {
    return 1;
  }

  // Don't draw backfaces
  const GLenum frontFace = GL_CCW;
  glFrontFace(frontFace);

  // Setup a framebuffer
  pangolin::GlTexture render(width, height);
  pangolin::GlRenderBuffer renderBuffer(width, height);
  pangolin::GlFramebuffer frameBuffer(render, renderBuffer);

  pangolin::GlTexture depthTexture(width, height, GL_R32F, false, 0, GL_RED,
                                   GL_FLOAT, 0);
  pangolin::GlFramebuffer depthFrameBuffer(depthTexture, renderBuffer);

  // Setup a camera
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrixRDF_BottomLeft(
          width, height, width / 2.0f, width / 2.0f, (width - 1.0f) / 2.0f,
          (height - 1.0f) / 2.0f, 0.1f, 100.0f),
      pangolin::ModelViewLookAtRDF(0, 0, 4, 0, 0, 0, 0, 1, 0));

  // Start at some origin
  Eigen::Matrix4d T_camera_world = s_cam.GetModelViewMatrix();

  // And move to the left
  Eigen::Matrix4d T_new_old = Eigen::Matrix4d::Identity();

  T_new_old.topRightCorner(3, 1) = Eigen::Vector3d(0.025, 0, 0);

  // load mirrors
  std::vector<MirrorSurface> mirrors;
  if (surfaceFile.length()) {
    std::ifstream file(surfaceFile);
    picojson::value json;
    picojson::parse(json, file);

    for (size_t i = 0; i < json.size(); i++) {
      mirrors.emplace_back(json[i]);
    }
    std::cout << "Loaded " << mirrors.size() << " mirrors" << std::endl;
  }

  const std::string shadir = STR(SHADER_DIR);
  MirrorRenderer mirrorRenderer(mirrors, width, height, shadir);

  // load mesh and textures
  PTexMesh ptexMesh(meshFile, atlasFolder);
  ptexMesh.SetExposure(0.01);
  ptexMesh.SetGamma(1.697);
  ptexMesh.SetSaturation(1.5);

  pangolin::ManagedImage<Eigen::Matrix<uint8_t, 3, 1>> image(width, height);
  pangolin::ManagedImage<float> depthImage(width, height);
  pangolin::ManagedImage<uint16_t> depthImageInt(width, height);

  // Render some frames
  size_t numFrames = 100;
  if (renderPoses.size() > 0) {
    numFrames = renderPoses.size();
  }

  auto output_path = file_path / "eval";
  std::filesystem::remove_all(output_path);
  std::filesystem::create_directories(output_path / "results");
  std::ofstream poseFile(output_path / "traj.txt");
  for (size_t i = 0; i < numFrames; i++) {
    std::cout << "\rRendering frame " << i + 1 << "/" << numFrames << "... ";
    std::cout.flush();

    if (renderPoses.size() > 0) {
      s_cam.GetModelViewMatrix() = renderPoses[i].Inverse();

      Eigen::Matrix4d view_matrix = renderPoses[i];
      // 16 col per line
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
          poseFile << view_matrix(j, k) << " ";
        }
      }
      poseFile << "\n";
    }

    // Render
    frameBuffer.Bind();
    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(0, 0, width, height);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glEnable(GL_CULL_FACE);

    ptexMesh.Render(s_cam);

    glDisable(GL_CULL_FACE);

    glPopAttrib(); // GL_VIEWPORT_BIT
    frameBuffer.Unbind();

    for (size_t i = 0; i < mirrors.size(); i++) {
      MirrorSurface &mirror = mirrors[i];
      // capture reflections
      mirrorRenderer.CaptureReflection(mirror, ptexMesh, s_cam, frontFace);

      frameBuffer.Bind();
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, width, height);

      // render mirror
      mirrorRenderer.Render(mirror, mirrorRenderer.GetMaskTexture(i), s_cam);

      glPopAttrib(); // GL_VIEWPORT_BIT
      frameBuffer.Unbind();
    }

    // Download and save
    render.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);

    char filename[1000];
    snprintf(filename, 1000, (output_path / "results/frame%06zu.jpg").c_str(),
             i);

    pangolin::SaveImage(image.UnsafeReinterpret<uint8_t>(),
                        pangolin::PixelFormatFromString("RGB24"),
                        std::string(filename));

    if (renderDepth) {
      // render depth
      depthFrameBuffer.Bind();
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, width, height);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

      glEnable(GL_CULL_FACE);

      ptexMesh.RenderDepth(s_cam, depthScale);

      glDisable(GL_CULL_FACE);

      glPopAttrib(); // GL_VIEWPORT_BIT
      depthFrameBuffer.Unbind();

      depthTexture.Download(depthImage.ptr, GL_RED, GL_FLOAT);

      // convert to 16-bit int
      for (size_t i = 0; i < depthImage.Area(); i++)
        depthImageInt[i] = static_cast<uint16_t>(depthImage[i] + 0.5f);

      snprintf(filename, 1000, (output_path / "results/depth%06zu.png").c_str(),
               i);
      pangolin::SaveImage(depthImageInt.UnsafeReinterpret<uint8_t>(),
                          pangolin::PixelFormatFromString("GRAY16LE"),
                          std::string(filename), true, 34.0f);
    }

    // Move the camera
    T_camera_world = T_camera_world * T_new_old.inverse();

    s_cam.GetModelViewMatrix() = T_camera_world;
  }
  std::cout << "\rRendering frame " << numFrames << "/" << numFrames
            << "... done" << std::endl;

  return 0;
}
