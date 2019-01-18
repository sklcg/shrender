#pragma once

#include <string>
#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/mesh.h>
#include <assimp/postprocess.h>

#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>

class Mesh{
private:
    std::vector<unsigned>    mGeometries;
    RTCDevice    device;
    RTCScene    scene;
public:
    // bounding box
    cv::Vec3f    bbox[2];
public:
    Mesh(){
        device = rtcNewDevice(NULL);

        if (rtcDeviceGetError(device) != RTC_NO_ERROR) {
            std::cout << "Failed to create device.\n";
            exit(-1);
        }

        scene = rtcDeviceNewScene(device, RTC_SCENE_STATIC, RTC_INTERSECT1);

        if (rtcDeviceGetError(device) != RTC_NO_ERROR) {
            std::cout << "Failed to create scene.\n";
            exit(-1);
        }
    }

    ~Mesh(){
        //// release
        //for (unsigned geomID : mGeometries)
        //{
        //    rtcDeleteGeometry(scene, geomID);
        //}

        //rtcDeleteScene(scene);
        //rtcDeleteDevice(device);
    }

    void Load(const std::string & filename){
        Assimp::Importer importer;
        const aiScene* ai_scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenSmoothNormals);
        if (!ai_scene) {
            printf("Unable to load mesh: %s\n", importer.GetErrorString());
            exit(1);
        }
        for (int i = 0; i < ai_scene->mNumMeshes; i++){
            aiMesh* mesh = ai_scene->mMeshes[0];
            unsigned geomID = rtcNewTriangleMesh2(scene, RTC_GEOMETRY_STATIC,
                mesh->mNumFaces, mesh->mNumVertices, 1);

            if (rtcDeviceGetError(device) != RTC_NO_ERROR) {
                std::cout << "Failed to create geom.\n";
                return exit(-1);
            }

            mGeometries.push_back(geomID);
            struct Vertex { float x, y, z, a; };
            struct Triangle { int v0, v1, v2; };

            Vertex* vertices = (Vertex*)rtcMapBuffer(scene, geomID, RTC_VERTEX_BUFFER);
            for (int j = 0; j < mesh->mNumVertices; j++){
                Vertex& v = vertices[j];
                aiVector3D& p = mesh->mVertices[j];
                v.x = p.x;
                v.y = p.y;
                v.z = p.z;
                v.a = 1.f;
            }
            rtcUnmapBuffer(scene, geomID, RTC_VERTEX_BUFFER);
            Triangle* triangles = (Triangle*)rtcMapBuffer(scene, geomID, RTC_INDEX_BUFFER);
            for (int j = 0; j < mesh->mNumFaces; j++){
                Triangle& t = triangles[j];
                aiFace& f = mesh->mFaces[j];
                t.v0 = f.mIndices[0];
                t.v1 = f.mIndices[1];
                t.v2 = f.mIndices[2];
            }
            rtcUnmapBuffer(scene, geomID, RTC_INDEX_BUFFER);
        }
        rtcCommit(scene);

        // get the bounding box
        RTCBounds bounds;
        rtcGetBounds(scene, bounds);

        bbox[0] = cv::Vec3f(bounds.lower_x, bounds.lower_y, bounds.lower_z);
        bbox[1] = cv::Vec3f(bounds.upper_x, bounds.upper_y, bounds.upper_z);
    }


    bool Intersect(RTCRay & ray) const{
        float t = ray.tfar;
        rtcIntersect(scene, ray);
        return ray.tfar < t;
    }

};