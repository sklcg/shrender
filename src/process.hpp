#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "util.hpp"
#include "light.hpp"
#include "mesh.hpp"
#include "render.hpp"

class Processor{
private:
    std::vector<std::string> _argv;

    cv::Size size;
    int spp;
    float fovX;
    float gamma;
    float exposure;
    bool background;

    int pHeight;
    int gaussianFilterSize;

    Mesh mesh;
    SHLight light;
    Camera camera;
    cv::Mat3f envmap;

    std::string savepath;

public:
    Processor() {
        // default configuration
        size = cv::Size(256, 256);
        spp = 1024;
        fovX = 60.f;
        gamma = 2.2f;
        exposure = 0.f;
        background = false;

        pHeight = 0;
        gaussianFilterSize = 1;
    }
    void setParameters(const std::string& name, const std::string& value) {
        if (name == "height") {
            size.height = std::stoi(value);
        }
        else if (name == "width") {
            size.width = std::stoi(value);
        }
        else if (name == "spp") {
            spp = std::stoi(value);
        }
        else if (name == "exposure") {
            exposure = std::stof(value);
        }
        else if (name == "gamma") {
            gamma = std::stof(value);
        }
        else if (name == "fov") {
            fovX = std::stof(value);
        }
        else if (name == "pheight") {
            pHeight = std::stoi(value);
        }
        else if (name == "gfSize") {
            gaussianFilterSize = std::stoi(value);
        }
        else {
            std::cout <<"No argument named "<< name << " type help for more hints"<< std::endl;
            exit(0);
        }
    }
    void DecodeArguments(int argc, char* argv[]) {
        int i = 0;
        while (i < argc) {
            if (argv[i][0] == '-' && argv[i][1] == '-' && i + 1 < argc) {
                // process optional arguments
                setParameters(argv[i] + 2, argv[i + 1]);
                i += 2;
            }
            else {
                // save indispensable arguments
                _argv.push_back(argv[i]);
                i += 1;
            }
        }
        if (_argv.size() > 1 && *_argv[1].rbegin() == '+') {
            background = true;
            _argv[1] = _argv[1].substr(0, _argv[1].size() - 1);
        }
        camera.lookAt = cv::Vec3f(0.f, 0.f, 0.f);
        camera.eye = cv::Vec3f(0.f, 0.f, 0.2f);
        camera.setFov(fovX, size.height * 1.0 / size.width);
    }
    void LoadData() {
        const std::string &t = _argv[1];

        // load sh light
        if (t == "shrender" || t == "sh2env") {
            light.GetFromFile(_argv[2]);
        }

        // load environment map
        if (t == "envrender" || t == "env2sh" || t == "eshrender" || t == "env2sh2env" || t == "panorama" || t == "view") {
            envmap = cv::imread(_argv[2], cv::IMREAD_UNCHANGED);
        }
        else if (t == "shrender" && background == true) {
            envmap = cv::imread(_argv[4], cv::IMREAD_UNCHANGED);
        }

        // load mesh
        if (t == "mask" || t == "shmap") {
            mesh.Load(_argv[2]);
        }
        else if (t == "envrender" || t == "shrender" || t == "eshrender") {
            mesh.Load(_argv[3]);
        }

        // read view direction x, y, z and rotate the envmap
        if (t == "env2sh" || t == "env2sh2env" || t == "panorama" || t == "view") {
            cv::Vec2f rotation = getRotation(cv::Vec3f(0, 0, -1), cv::Vec3f(std::stof(_argv[3]), std::stof(_argv[4]), std::stof(_argv[5])));
            envmap = rotate(envmap, rotation[0], rotation[1]);
        }
        else if (t == "envrender" || t == "eshrender") {
            cv::Vec2f rotation = getRotation(cv::Vec3f(0, 0, -1), cv::Vec3f(std::stof(_argv[4]), std::stof(_argv[5]), std::stof(_argv[6])));
            envmap = rotate(envmap, rotation[0], rotation[1]);
        }
        else if (t == "shrender" && background) {
            cv::Vec2f rotation = getRotation(cv::Vec3f(0, 0, -1), cv::Vec3f(std::stof(_argv[5]), std::stof(_argv[6]), std::stof(_argv[7])));
            envmap = rotate(envmap, rotation[0], rotation[1]);
        }
        savepath = *_argv.rbegin();
    }
    void _Process() {
        if (_argv.size() == 8 && _argv[1] == "envrender")
            RenderMeshUsingEnvmap();
        else if ((_argv.size() == 5 || _argv.size() == 9) && _argv[1] == "shrender")
            RenderMeshUsingSH();
        else if (_argv.size() == 8 && _argv[1] == "eshrender")
            RenderMeshUsingEnvmapAfterSHProjection();
        else if (_argv.size() == 4 && _argv[1] == "sh2env")
            RenderEnvmapUsingSH();
        else if (_argv.size() == 7 && _argv[1] == "env2sh")
            RenderSHUsingEnvmap();
        else if (_argv.size() == 7 && _argv[1] == "env2sh2env")
            RenderEnvmapUsingEnvmapAfterSHProjection();
        else if (_argv.size() == 7 && _argv[1] == "panorama")
            Panorama();
        else if (_argv.size() == 7 && _argv[1] == "view")
            View();
        else if (_argv.size() == 4 && _argv[1] == "mask")
            Mask();
        else if (_argv.size() == 4 && _argv[1] == "shmap")
            RenderSHmapFromMesh();
        else if (_argv.size() == 2 && _argv[1] == "help")
            ShowUsage();
        else
            std::cout << "Invalid command arguments, type \"" << _argv[0] << " help\" for usages" << std::endl;
    }
    void Process(int argc, char* argv[]) {
        DecodeArguments(argc, argv);
        if (_argv.size() <= 1) {
            std::cout << "type \""<<_argv[0]<<" help\" for usages" << std::endl;
        }
        else {
            LoadData();
            _Process();
        }
    }
    void RenderMeshUsingEnvmap() {
        Renderer renderer(spp);
        cv::Mat3f image = renderer.Render(mesh, camera, envmap, size, background);
        WriteImage(savepath, image, "LDR");
    }
    void RenderMeshUsingSH() {
        Renderer renderer(spp);
        cv::Mat3f image = renderer.Render(mesh, camera, light, size, background ? envmap : cv::Mat3f(0,0));
        WriteImage(savepath, image, "LDR");
    }
    void RenderMeshUsingEnvmapAfterSHProjection() {
        Renderer renderer(spp);
        light.GetFromEnvmap(envmap);
        cv::Mat3f image = renderer.Render(mesh, camera, light, size, background ? envmap : cv::Mat3f(0, 0));
        WriteImage(savepath, image, "LDR");
    }
    void RenderSHUsingEnvmap() {
        Renderer renderer(spp);
        light.GetFromEnvmap(envmap);
        light.Write(savepath);
    }
    void RenderEnvmapUsingSH() {
        Renderer renderer(spp);
        cv::Mat3f image = light.RenderEnvmap(size.width, size.height);
        WriteImage(savepath, image, "HDR");
    }
    void RenderEnvmapUsingEnvmapAfterSHProjection() {
        Renderer renderer(spp);
        light.GetFromEnvmap(envmap);

        cv::Mat3f image = light.RenderEnvmap(size.width, size.height);
        WriteImage(savepath, image, "HDR");
    }
    void Panorama() {
        if (pHeight > 0) {
            cv::resize(envmap, envmap, cv::Size(pHeight * 2, pHeight));
        }
        WriteImage(savepath, envmap, "HDR");
    }
    void View() {
        Renderer renderer(spp);
        cv::Mat3f image = renderer.View(camera, envmap, size);
        WriteImage(savepath, image, "LDR");
    }
    void Mask() {
        Renderer renderer(spp);
        cv::Mat3f image = renderer.Mask(mesh, camera, size);
        WriteImage(savepath, image, "LDR");
    }
    void RenderSHmapFromMesh(){
        Renderer renderer(spp);
        SHMap shmap = renderer.RenderSH(mesh,camera,size);
        WriteSHMap(savepath, shmap);
    }
    void ShowUsage() {
        std::cout << "Please refer to theses usages:" << std::endl;
        std::cout << "\n----------" << "Indispensable Parameters" << "----------\n" << std::endl;
        std::cout << "\t" << _argv[0] << " envrender  envmap  mesh              x y z  output" << std::endl;
        std::cout << "\t" << _argv[0] << " envrender+ envmap  mesh              x y z  output" << std::endl;

        std::cout << "\t" << _argv[0] << "  shrender  shfile  mesh                     output" << std::endl;
        std::cout << "\t" << _argv[0] << "  shrender+ shfile  mesh  background  x y z  output" << std::endl;

        std::cout << "\t" << _argv[0] << " eshrender  envmap  mesh              x y z  output" << std::endl;
        std::cout << "\t" << _argv[0] << " eshrender+ envmap  mesh              x y z  output" << std::endl;

        std::cout << "\t" << _argv[0] << "    env2sh  envmap                    x y z  output" << std::endl;
        std::cout << "\t" << _argv[0] << "    sh2env  shfile                           output" << std::endl;
        std::cout << "\t" << _argv[0] << " env2sh2env envmap                    x y z  output" << std::endl;

        std::cout << "\t" << _argv[0] << "   panorama envmap                    x y z  output" << std::endl;
        std::cout << "\t" << _argv[0] << "       view envmap                    x y z  output" << std::endl;
        std::cout << "\t" << _argv[0] << "       mask         mesh                     output" << std::endl;
        std::cout << "\t" << _argv[0] << "      shmap         mesh                     output" << std::endl;

        std::cout << "\t" << "+ for rendering result with background" << std::endl;

        std::cout << "\n----------" << "Optional Parameters" << "----------\n" << std::endl;
        std::cout << "\t" << " --spp      : " << "Default(1024), Samples per pixel during rendering" << std::endl;
        std::cout << "\t" << " --fov      : " << "Default(60.0), Field Of View (FOV) on horizontal (degree)" << std::endl;
        std::cout << "\t" << " --width    : " << "Default( 256), Width of the output image" << std::endl;
        std::cout << "\t" << " --height   : " << "Default( 256), Height of the output image" << std::endl;
        std::cout << "\t" << " --exposure : " << "Default( 0.0), Scale the values of output image by 2^exposure" << std::endl;
        std::cout << "\t" << " --gamma    : " << "Default( 2.2), The gamma curve applied to the output image" << std::endl;

        std::cout << "\n----------" << "Examples" << "----------\n" << std::endl;
        std::cout << "\t" << "illumination envrender+ office.hdr d:/meshs/bunny.ply office.hdr 0 0 1 output.png --spp 512 --fov 90" << std::endl;
        std::cout << "\t" << "illumination view  example.hdr 0 0 1 output.png --exposure 2.0" << std::endl;
    }
    void WriteSHMap(std::string savepath, SHMap shmap, int num_coeff = 16){
        std::ofstream fout(savepath, std::ios::out);
        fout << size.height << " " << size.width << " " << num_coeff << std::endl;
        for (int i = 0; i < shmap.size(); ++i) {
            fout << shmap[i].size();
            for (int j = 0; j < shmap[i].size(); ++j) {
                fout << " " << shmap[i][j];
            }
            fout << std::endl;
        }
        fout.close();
    };
    void WriteImage(const std::string& filename, const cv::Mat3f& image, const std::string &defaultFormat = "LDR") {
        std::string postfix = getPostfix(filename);
        cv::Mat3f ConvertedImage = image * cv::pow(2, exposure);
        if (gaussianFilterSize > 0) {
            cv::GaussianBlur(ConvertedImage, ConvertedImage, cv::Size(3, 3), 0, 0);
        }
        if (postfix == "hdr" || postfix == "exr") {
            cv::imwrite(filename, ConvertedImage);
        }
        else if (postfix == "jpg" || postfix == "jpeg" || postfix == "jpe"
            || postfix == "dib" || postfix == "png" || postfix == "bmp"
            || postfix == "jp2" || postfix == "webp" || postfix == "pbm"
            || postfix == "pgm" || postfix == "ppm" || postfix == "pnm"
            || postfix == "pxm" || postfix == "tiff" || postfix == "tif"
            ) {
            cv::pow(ConvertedImage, 1 / gamma, ConvertedImage);
            cv::imwrite(filename, ConvertedImage * 255.f);
        }
        else if (defaultFormat == "HDR") {
            cv::imwrite(filename + ".hdr", ConvertedImage);
        }
        else {
            cv::pow(ConvertedImage, 1 / gamma, ConvertedImage);
            cv::imwrite(filename + ".png", ConvertedImage * 255.f);
        }
    }
    std::string getPostfix(const std::string &str) {
        std::string lower(str);
        std::transform(str.begin(), str.end(), lower.begin(), ::tolower);
        int index = lower.rfind('.');
        if (index == -1) {
            return std::string("");
        }
        return lower.substr(index + 1);
    }
};