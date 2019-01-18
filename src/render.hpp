#pragma once
#include <opencv2/opencv.hpp>

#include "util.hpp"
#include "mesh.hpp"
#include "light.hpp"

typedef std::vector<std::vector<float>> SHMap;

class Camera
{
public:
	cv::Vec3f lookAt;
	cv::Vec3f eye;
	cv::Vec3f up;

	float fov;
	float fovX;
	float fovY;

	Camera::Camera() {
		this->up = cv::Vec3f(0.f, 1.f, 0.f);
		this->setFov(60.f, 1.0);
	}

	void Camera::setFov(float fov, float ratio) {
		this->fovX = fov;
		this->fovY = atan(tan(fov * M_PI / 360) * ratio) * 360 / M_PI;
	}
};
class Renderer
{
private:
	int n_samples;
	// precompute samples
	std::vector<cv::Vec3f>			samples;
	std::vector<std::vector<float>> sample_coeffs;
	std::vector<float>				sample_weights;

public:
	Renderer(unsigned int n) : n_samples(n)
	{
		cv::RNG random;
		samples.resize(n_samples);
		sample_coeffs.resize(n_samples);
		sample_weights.reserve(n_samples);
		float oneoverN = 1.f / sqrt(n_samples);
		for (int i = 0; i < n_samples; i++)
		{
			float u = cv::min(halton(i, 2) + random.uniform(0.f, 1.f)*oneoverN, 1.f);
			float v = cv::min(halton(i, 3) + random.uniform(0.f, 1.f)*oneoverN, 1.f);

			// sample direction
			samples[i] = uniform_mapping(u, v);

			// precompute SH coefficnents for sample directions
			cv::Vec2f sphere = to_sphere(samples[i]);
			sample_coeffs[i] = EvalSH(sphere[0], sphere[1]);

			sample_weights[i] = 1.f / n_samples;
		}
	}
	
	RTCRay GetRTCRay(const cv::Vec3f& o, const cv::Vec3f& d)
	{
		RTCRay ray;
		ray.org[0] = o[0];
		ray.org[1] = o[1];
		ray.org[2] = o[2];

		ray.dir[0] = d[0];
		ray.dir[1] = d[1];
		ray.dir[2] = d[2];

		ray.tnear = FLT_EPSILON;
		ray.tfar = FLT_MAX;

		return ray;
	}
	cv::Mat3f Render(const Mesh & mesh, const Camera & cam, const cv::Mat3f& envmap, cv::Size size, bool background = false)
	{
		cv::Mat3f img = cv::Mat3f::zeros(size);

		// camera
		cv::Vec3f dir0 = cv::normalize(cam.lookAt - cam.eye);
		cv::Vec3f right = dir0.cross(cam.up);
		cv::Vec3f up = right.cross(dir0);

		float ry = tanf(cam.fovY / 360.f * M_PI);
		float rx = tanf(cam.fovX / 360.f * M_PI);

		right = cv::normalize(right) * 2.0 * rx;
		up = cv::normalize(up) * 2.0 * ry;

		#pragma omp parallel for
		for (int x = 0; x < img.cols; x++)
		{
			for (int y = 0; y < img.rows; y++)
			{
				float u0 = (float)(x + 0.5f) / img.cols;
				float v0 = (float)(y + 0.5f) / img.rows;

				cv::Vec3f dir = cv::normalize(dir0 + (u0 - 0.5f) * right - (v0 - 0.5f) * up);

				RTCRay ray = GetRTCRay(cam.eye, dir);

				if (mesh.Intersect(ray))
				{
					cv::Vec3f normal = cv::normalize(cv::Vec3f(ray.Ng));

					// flip normal
					if (normal.dot(dir) > 0) normal = -normal;

					// intersection point
					cv::Vec3f position = cam.eye + ray.tfar * dir + (normal * 1e-5);

					// sample envmap
					for (int i = 0; i < n_samples; i++)
					{
						cv::Vec3f sample = samples[i];

						RTCRay sample_ray = GetRTCRay(position, sample);

						if (!mesh.Intersect(sample_ray))
						{
							// N dot L
							float nDotL = cv::max(0.f, normal.dot(sample));

							// lookup envmap
							float theta = acosf(sample[1]) / M_PI;
							float phi = atan2f(-sample[0], sample[2]) / M_PI / 2.f + 0.5;

							// sample environment map
							cv::Vec3f vec = lookup(envmap, phi * envmap.cols, theta * envmap.rows);

							// pixel color
							img(y, x) += sample_weights[i] * nDotL * vec;
						}
					}
				}
				else
				{
					if (background) {
						cv::Vec2f sphere_coord = to_sphere(dir);
						float u = sphere_coord[1] / M_PI / 2.0;
						float v = sphere_coord[0] / M_PI;
						img(y, x) = lookup(envmap, u * envmap.cols, v * envmap.rows);
					}
					else{
						img(y, x) = cv::Vec3f(0.f, 0.f, 0.f);
					}
				}
			}
		}

		return img;
	}

	cv::Mat3f Render(const Mesh & mesh, const Camera & cam, const SHLight & light, cv::Size size, const cv::Mat3f& background = cv::Mat3f(0,0))
	{
		SHMap coeff = RenderSH(mesh, cam, size, background);
		return RenderSH(coeff, light, size);
	}

	SHMap RenderSH(const Mesh & mesh, const Camera & cam, cv::Size size, const cv::Mat3f& background = cv::Mat3f(0, 0))
	{
		SHMap coeff_map(size.area());

		// camera
		cv::Vec3f dir0 = cv::normalize(cam.lookAt - cam.eye);
		cv::Vec3f right = dir0.cross(cam.up);
		cv::Vec3f up = right.cross(dir0);

		float ry = tanf(cam.fovY / 360.f * M_PI);
		float rx = tanf(cam.fovX / 360.f * M_PI);

		right = cv::normalize(right) * 2.0 * rx;
		up = cv::normalize(up) * 2.0 * ry;

		#pragma omp parallel for
		for (int x = 0; x < size.width; x++)
		{
			for (int y = 0; y < size.height; y++)
			{
				float u0 = (float)(x + 0.5f) / size.width;
				float v0 = (float)(y + 0.5f) / size.height;

				cv::Vec3f dir = cv::normalize(dir0 + (u0 - 0.5f) * right - (v0 - 0.5f) * up);

				RTCRay ray = GetRTCRay(cam.eye, dir);

				std::vector<float> coeff;

				if (mesh.Intersect(ray))
				{
					coeff.resize(16, 0.f);

					cv::Vec3f normal = cv::normalize(cv::Vec3f(ray.Ng));

					// flip normal
					if (normal.dot(dir) > 0) normal = -normal;

					// intersection point
					cv::Vec3f position = cam.eye + ray.tfar * dir + (normal * 1e-5);

					// sample envmap
					for (int i = 0; i < n_samples; i++)
					{
						cv::Vec3f sample = samples[i];

						RTCRay sample_ray = GetRTCRay(position, sample);

						if (!mesh.Intersect(sample_ray))
						{
							// N dot L
							float nDotL = cv::max(0.f, normal.dot(sample));

							for (int k = 0; k < coeff.size(); k++)
							{
								coeff[k] += sample_coeffs[i][k] * sample_weights[i] * nDotL;
							}
						}
					}
				}
				else
				{
					coeff.clear();
					if (background.size().height != 0) {
						cv::Vec2f sphere_coord = to_sphere(dir);
						float u = sphere_coord[1] / M_PI / 2.0;
						float v = sphere_coord[0] / M_PI;

						cv::Vec3f color = lookup(background, u * background.cols, v * background.rows);
						coeff.push_back(color[0]);
						coeff.push_back(color[1]);
						coeff.push_back(color[2]);
					}
				}
				coeff_map[y * size.width + x] = coeff;
			}
		}

		return coeff_map;
	}

	cv::Mat3f RenderSH(const SHMap & map, const SHLight & light, cv::Size size)
	{
		cv::Mat3f img(size);

		for (int x = 0; x < img.cols; x++)
		{
			for (int y = 0; y < img.rows; y++)
			{
				std::vector<float> coeff = map[y * size.width + x];
				if ((int)coeff.size() == 3) {
					// background color
					img(y, x) = cv::Vec3f(coeff[0], coeff[1], coeff[2]);
				}
				else {
					// render by sh coefficients;
					img(y, x) = light.Render(coeff);
				}
			}
		}
		return img;
	}
	cv::Mat3f Mask(const Mesh & mesh, const Camera &cam, cv::Size size)
	{
		cv::Mat3f img = cv::Mat3f::zeros(size);

		cv::Vec3f dir0 = cv::normalize(cam.lookAt - cam.eye);
		cv::Vec3f right = dir0.cross(cam.up);
		cv::Vec3f up = right.cross(dir0);

		float ry = tanf(cam.fovY / 360.f * M_PI);
		float rx = tanf(cam.fovX / 360.f * M_PI);

		right = cv::normalize(right) * 2.0 * rx;
		up = cv::normalize(up) * 2.0 * ry;

		#pragma omp parallel for
		for (int x = 0; x < size.width; x++)
		{
			for (int y = 0; y < size.height; y++)
			{
				float u0 = (float)(x + 0.5f) / size.width;
				float v0 = (float)(y + 0.5f) / size.height;

				cv::Vec3f dir = cv::normalize(dir0 + (u0 - 0.5f) * right - (v0 - 0.5f) * up);

				RTCRay ray = GetRTCRay(cam.eye, dir);

				if (mesh.Intersect(ray)) {
					img(y, x) = cv::Vec3f(1.f, 1.f, 1.f);
				}
			}
		}
		return img;
	}

	cv::Mat3f View(const Camera& cam, const cv::Mat3f& envmap, cv::Size size) {

		cv::Mat3f img = cv::Mat3f::zeros(size);

		// camera
		cv::Vec3f dir0 = cv::normalize(cam.lookAt - cam.eye);
		cv::Vec3f right = dir0.cross(cam.up);
		cv::Vec3f up = right.cross(dir0);

		float ry = tanf(cam.fovY / 360.f * M_PI);
		float rx = tanf(cam.fovX / 360.f * M_PI);

		right = cv::normalize(right) * 2.0 * rx;
		up = cv::normalize(up) * 2.0 * ry;

		#pragma omp parallel for
		for (int x = 0; x < img.cols; x++)
		{
			for (int y = 0; y < img.rows; y++)
			{

				float u0 = (float)(x + 0.5f) / img.cols;
				float v0 = (float)(y + 0.5f) / img.rows;

				cv::Vec3f dir = cv::normalize(dir0 + (u0 - 0.5f) * right - (v0 - 0.5f) * up);

				cv::Vec2f sphere_coord = to_sphere(dir);

				float u = sphere_coord[1] / M_PI / 2.0 * envmap.cols;
				float v = sphere_coord[0] / M_PI * envmap.rows;

				img(y, x) += lookup(envmap, u, v);
			}
		}
		return img;
	}
};