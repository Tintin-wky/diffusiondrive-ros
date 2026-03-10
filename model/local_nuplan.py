from __future__ import annotations

from enum import Enum, IntEnum
import math
import numpy as np
import numpy.typing as npt
from io import BytesIO
from dataclasses import dataclass
from typing import Optional, cast, Iterable, List, Union, IO, Any, NamedTuple, ByteString, Dict, Tuple, Set
import pytest
import colorsys
from matplotlib import axes, cm
from PIL import Image
from pyquaternion import Quaternion
PROXIMITY_ABS_TOL = 1e-10


class PointCloudHeader(NamedTuple):
    """Class for Point Cloud header."""

    version: str
    fields: List[str]
    size: List[int]
    type: List[str]
    count: List[int]  # type: ignore
    width: int
    height: int
    viewpoint: List[int]
    points: int
    data: str


class PointCloud:
    """
    Class for raw .pcd file.
    """

    def __init__(self, header: PointCloudHeader, points: npt.NDArray[np.float64]) -> None:
        """
        PointCloud.
        :param header: Pointcloud header.
        :param points: <np.ndarray, X, N>. X columns, N points.
        """
        self._header = header
        self._points = points

    @property
    def header(self) -> PointCloudHeader:
        """
        Returns pointcloud header.
        :return: A PointCloudHeader instance.
        """
        return self._header

    @property
    def points(self) -> npt.NDArray[np.float64]:
        """
        Returns points.
        :return: <np.ndarray, X, N>. X columns, N points.
        """
        return self._points

    def save(self, file_path: str) -> None:
        """
        Saves to .pcd file.
        :param file_path: The path to the .pcd file.
        """
        with open(file_path, 'wb') as fp:
            fp.write('# .PCD v{} - Point Cloud Data file format\n'.format(self._header.version).encode('utf8'))
            for field in self._header._fields:
                value = getattr(self._header, field)
                if isinstance(value, list):
                    text = ' '.join(map(str, value))
                else:
                    text = str(value)
                fp.write('{} {}\n'.format(field.upper(), text).encode('utf8'))
            fp.write(self._points.tobytes())

    @classmethod
    def parse(cls, pcd_content: bytes) -> PointCloud:
        """
        Parses the pointcloud from byte stream.
        :param pcd_content: The byte stream that holds the pcd content.
        :return: A PointCloud object.
        """
        with BytesIO(pcd_content) as stream:
            header = cls.parse_header(stream)
            points = cls.parse_points(stream, header)
            return cls(header, points)

    @classmethod
    def parse_from_file(cls, pcd_file: str) -> PointCloud:
        """
        Parses the pointcloud from .pcd file on disk.
        :param pcd_file: The path to the .pcd file.
        :return: A PointCloud instance.
        """
        with open(pcd_file, 'rb') as stream:
            header = cls.parse_header(stream)
            points = cls.parse_points(stream, header)
            return cls(header, points)

    @staticmethod
    def parse_header(stream: IO[Any]) -> PointCloudHeader:
        """
        Parses the header of a pointcloud from byte IO stream.
        :param stream: Binary stream.
        :return: A PointCloudHeader instance.
        """
        headers_list = []
        while True:
            line = stream.readline().decode('utf8').strip()
            if line.startswith('#'):
                continue
            columns = line.split()
            key = columns[0].lower()
            val = columns[1:] if len(columns) > 2 else columns[1]
            headers_list.append((key, val))

            if key == 'data':
                break

        headers = dict(headers_list)
        headers['size'] = list(map(int, headers['size']))
        headers['count'] = list(map(int, headers['count']))
        headers['width'] = int(headers['width'])
        headers['height'] = int(headers['height'])
        headers['viewpoint'] = list(map(int, headers['viewpoint']))
        headers['points'] = int(headers['points'])
        header = PointCloudHeader(**headers)

        if any([c != 1 for c in header.count]):
            raise RuntimeError('"count" has to be 1')

        if not len(header.fields) == len(header.size) == len(header.type) == len(header.count):
            raise RuntimeError('fields/size/type/count field number are inconsistent')

        return header

    @staticmethod
    def parse_points(stream: IO[Any], header: PointCloudHeader) -> npt.NDArray[np.float64]:
        """
        Parses points from byte IO stream.
        :param stream: Byte stream that holds the points.
        :param header: <np.ndarray, X, N>. A numpy array that has X columns(features), N points.
        :return: Points of Point Cloud.
        """
        if header.data != 'binary':
            raise RuntimeError('Un-supported data foramt: {}. "binary" is expected.'.format(header.data))

        # There is garbage data at the end of the stream, usually all b'\x00'.
        row_type = PointCloud.np_type(header)
        length = row_type.itemsize * header.points
        buff = stream.read(length)
        if len(buff) != length:
            raise RuntimeError('Incomplete pointcloud stream: {} bytes expected, {} got'.format(length, len(buff)))

        points = np.frombuffer(buff, row_type)

        return points

    @staticmethod
    def np_type(header: PointCloudHeader) -> np.dtype:  # type: ignore
        """
        Helper function that translate column types in pointcloud to np types.
        :param header: A PointCloudHeader object.
        :return: np.dtype that holds the X features.
        """
        type_mapping = {'I': 'int', 'U': 'uint', 'F': 'float'}
        np_types = [type_mapping[t] + str(int(s) * 8) for t, s in zip(header.type, header.size)]

        return np.dtype([(f, getattr(np, nt)) for f, nt in zip(header.fields, np_types)])

    def to_pcd_bin(self) -> npt.NDArray[np.float32]:
        """
        Converts pointcloud to .pcd.bin format.
        :return: <np.float32, 5, N>, the point cloud in .pcd.bin format.
        """
        lidar_fields = ['x', 'y', 'z', 'intensity', 'ring']
        return np.array([np.array(self.points[f], dtype=np.float32) for f in lidar_fields])

    def to_pcd_bin2(self) -> npt.NDArray[np.float32]:
        """
        Converts pointcloud to .pcd.bin2 format.
        :return: <np.float32, 6, N>, the point cloud in .pcd.bin2 format.
        """
        lidar_fields = ['x', 'y', 'z', 'intensity', 'ring', 'lidar_info']
        return np.array([np.array(self.points[f], dtype=np.float32) for f in lidar_fields])

def view_points(
    points: npt.NDArray[np.float64], view: npt.NDArray[np.float64], normalize: bool
) -> npt.NDArray[np.float64]:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

def rainbow(nbr_colors: int, normalized: bool = False) -> List[Tuple[Any, ...]]:
    """
    Returns colors that are maximally different in HSV color space.
    :param nbr_colors: Number of colors to generate.
    :param normalized: Whether to normalize colors in 0-1. Else it is between 0-255.
    :return: <[(R <TYPE>, G <TYPE>, B <TYPE>)]>. Color <TYPE> varies depending on whether they are normalized.
    """
    hsv_tuples = [(x * 1.0 / nbr_colors, 0.5, 1) for x in range(nbr_colors)]
    colors = 255 * np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)))
    if normalized:
        colors = colors / 255.0  # type: ignore
        return list(colors)
    else:
        return [tuple([int(c) for c in color]) for color in colors]

# Field name used to find lidar sweep time data in pcd files.
PCD_TIMESTAMP_FIELD_NAME = 'time_delta'


def pcd_to_numpy(pcd_file: str) -> npt.NDArray[np.float32]:
    """
    This function converts the pointcloud *.pcl or *.pcd files to numpy (x, y, z, i) format,
    or (x, y, z, i, t) format if a time field is present.
    :param pcd_file: Name of the point cloud file (*.pcl or *.pcd)
    :return: A numpy array of shape (n, 4) or (n, 5), dtype = np.float32
    """
    with open(pcd_file) as ifile:
        data = [line.strip() for line in ifile]

    meta = data[:10]
    assert meta[0].startswith('#'), 'First line must be comment'
    assert meta[1].startswith('VERSION'), 'Second line must be VERSION'

    # FIELDS is the third line in the file. Remove the word FIELDS so the first field name is at index 0 in this list.
    fields = meta[2].split(' ')[1:]
    assert all(f in fields for f in ['x', 'y', 'z']), 'x, y, and z fields are required'

    # not support binary.
    assert data[10] == 'DATA ascii'

    data = data[11:]  # remove header stuff
    data = [d.split(' ') for d in data]  # type: ignore  # split each line

    # Currently we ignore the SIZE and TYPE lines in pcd files and assume everything's a 32-bit float.
    all_columns = np.array(data, dtype=np.float32)  # type: ignore
    num_points = all_columns.shape[0]
    has_delta_time = PCD_TIMESTAMP_FIELD_NAME in fields
    result_shape = (num_points, 5) if has_delta_time else (num_points, 4)
    result = np.zeros(result_shape, dtype=np.float32)  # type: ignore

    result[:, 0] = all_columns[:, fields.index('x')]
    result[:, 1] = all_columns[:, fields.index('y')]
    result[:, 2] = all_columns[:, fields.index('z')]
    if 'intensity' in fields:
        result[:, 3] = all_columns[:, fields.index('intensity')]
    if has_delta_time:
        result[:, 4] = all_columns[:, fields.index(PCD_TIMESTAMP_FIELD_NAME)]

    return result


class LidarPointCloud:
    """Simple data class representing a point cloud."""

    def __init__(self, points: npt.NDArray[np.float32]) -> None:
        """
        Class for manipulating and viewing point clouds.
        :param points: <np.float: f, n>. Input point cloud matrix with f features per point and n points.
        """
        if points.ndim == 1:
            points = np.atleast_2d(points).T

        self.points = points

    @staticmethod
    def load_pcd_bin(pcd_bin: Union[str, IO[Any], ByteString], pcd_bin_version: int = 1) -> npt.NDArray[np.float32]:
        """
        Loads from pcd binary format:
            version 1: a numpy array with 5 cols (x, y, z, intensity, ring).
            version 2: a numpy array with 6 cols (x, y, z, intensity, ring, lidar_id).
        :param pcd_bin: File path or a file-like object or raw bytes.
        :param pcd_bin_version: 1 or 2, see above.
        :return: <np.float: 6, n>. Point cloud matrix[(x, y, z, intensity, ring, lidar_id)].
        """
        if isinstance(pcd_bin, str):
            scan = np.fromfile(pcd_bin, dtype=np.float32)  # type: ignore
        else:
            if not isinstance(pcd_bin, bytes):
                pcd_bin = pcd_bin.read()  # type: ignore
            scan = np.frombuffer(pcd_bin, dtype=np.float32)  # type: ignore
            # frombuffer returns a read-only np.array
            scan = np.copy(scan)

        if pcd_bin_version == 1:
            points = scan.reshape((-1, 5))
            # Append lidar_id column
            points = np.hstack((points, -1 * np.ones((points.shape[0], 1), dtype=np.float32)))
        elif pcd_bin_version == 2:
            points = scan.reshape((-1, 6))
        else:
            pytest.fail('Unknown pcd bin file version: %d' % pcd_bin_version)

        return points.T

    @staticmethod
    def load_pcd(pcd_data: Union[IO[Any], ByteString]) -> npt.NDArray[np.float32]:
        """
        Loads a pcd file.
        :param pcd_data: File path or a file-like object or raw bytes.
        :return: <np.float: 6, n>. Point cloud matrix[(x, y, z, intensity, ring, lidar_id)].
        """
        if not isinstance(pcd_data, bytes):
            pcd_data = pcd_data.read()  # type: ignore

        return PointCloud.parse(pcd_data).to_pcd_bin2()  # type: ignore

    @classmethod
    def from_file(cls, file_name: str) -> LidarPointCloud:
        """
        Instantiates from a .pcl, .pcd, .npy, or .bin file.
        :param file_name: Path of the pointcloud file on disk.
        :return: A LidarPointCloud object.
        """
        if file_name.endswith('.bin'):
            points = cls.load_pcd_bin(file_name, 1)
        elif file_name.endswith('.bin2'):
            points = cls.load_pcd_bin(file_name, 2)
        elif file_name.endswith('.pcl') or file_name.endswith('.pcd'):
            points = pcd_to_numpy(file_name).T
        elif file_name.endswith('.npy'):
            points = np.load(file_name)
        else:
            raise ValueError('Unsupported filetype {}'.format(file_name))

        return cls(points)

    @classmethod
    def from_buffer(cls, pcd_data: Union[IO[Any], ByteString], content_type: str = 'bin') -> LidarPointCloud:
        """
        Instantiates from buffer.
        :param pcd_data: File path or a file-like object or raw bytes.
        :param content_type: Type of the point cloud content, such as 'bin', 'bin2', 'pcd'.
        :return: A LidarPointCloud object.
        """
        if content_type == 'bin':
            return cls(cls.load_pcd_bin(pcd_data, 1))
        elif content_type == 'bin2':
            return cls(cls.load_pcd_bin(pcd_data, 2))
        elif content_type == 'pcd':
            return cls(cls.load_pcd(pcd_data))
        else:
            raise NotImplementedError('Not implemented content type: %s' % content_type)

    @classmethod
    def make_random(cls) -> LidarPointCloud:
        """
        Instantiates a random point cloud.
        :return: LidarPointCloud instance.
        """
        return LidarPointCloud(points=np.random.normal(0, 100, size=(4, 100)))  # type: ignore

    def __eq__(self, other: object) -> bool:
        """
        Checks if two LidarPointCloud are equal.
        :param other: Other object.
        :return: True if both objects are equal otherwise False.
        """
        if not isinstance(other, LidarPointCloud):
            return NotImplemented

        return np.allclose(self.points, other.points, atol=1e-06)

    def copy(self) -> LidarPointCloud:
        """
        Creates a copy of self.
        :return: LidarPointCloud instance.
        """
        return LidarPointCloud(points=self.points.copy())

    def nbr_points(self) -> int:
        """
        Returns the number of points.
        :return: Number of points.
        """
        return int(self.points.shape[1])

    def subsample(self, ratio: float) -> None:
        """
        Sub-samples the pointcloud.
        :param ratio: Fraction to keep.
        """
        assert 0 < ratio < 1

        selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, min_dist: float) -> None:
        """
        Removes points too close within a certain distance from origin from bird view (so dist = sqrt(x^2+y^2)).
        :param min_dist: The distance threshold.
        """
        dist_from_orig = np.linalg.norm(self.points[:2, :], axis=0)
        self.points = self.points[:, dist_from_orig >= min_dist]

    def radius_filter(self, radius: float) -> None:
        """
        Removes points outside the given radius.
        :param radius: Radius in meters.
        """
        keep = np.sqrt(self.points[0] ** 2 + self.points[1] ** 2) <= radius
        self.points = self.points[:, keep]

    def range_filter(
        self,
        xrange: Tuple[float, float] = (-np.inf, np.inf),
        yrange: Tuple[float, float] = (-np.inf, np.inf),
        zrange: Tuple[float, float] = (-np.inf, np.inf),
    ) -> None:
        """
        Restricts points to specified ranges.
        :param xrange: (xmin, xmax).
        :param yrange: (ymin, ymax).
        :param zrange: (zmin, zmax).
        """
        # Figure out which points to keep.
        keep_x = np.logical_and(xrange[0] <= self.points[0], self.points[0] <= xrange[1])
        keep_y = np.logical_and(yrange[0] <= self.points[1], self.points[1] <= yrange[1])
        keep_z = np.logical_and(zrange[0] <= self.points[2], self.points[2] <= zrange[1])
        keep = np.logical_and(keep_x, np.logical_and(keep_y, keep_z))

        self.points = self.points[:, keep]

    def translate(self, x: npt.NDArray[np.float64]) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3,>. Translation in x, y, z.
        """
        self.points[:3] += x.reshape((-1, 1))

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Applies a rotation.
        :param quaternion: Rotation to apply.
        """
        self.points[:3] = np.dot(quaternion.rotation_matrix.astype(np.float32), self.points[:3])

    def transform(self, transf_matrix: npt.NDArray[np.float64]) -> None:
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogeneous transformation matrix.
        """
        transf_matrix = transf_matrix.astype(np.float32)
        self.points[:3, :] = transf_matrix[:3, :3] @ self.points[:3] + transf_matrix[:3, 3].reshape((-1, 1))

    def scale(self, scale: Tuple[float, float, float]) -> None:
        """
        Scales the lidar xyz coordinates.
        :param scale: The scaling parameter.
        """
        scale_arr = np.array(scale)  # type: ignore
        scale_arr.shape = (3, 1)  # Make sure it is a column vector.
        self.points[:3, :] *= np.tile(scale_arr, (1, self.nbr_points()))

    def render_image(
        self,
        canvas_size: Tuple[int, int] = (1001, 1001),
        view: npt.NDArray[np.float64] = np.array([[10, 0, 0, 500], [0, 10, 0, 500], [0, 0, 10, 0]]),
        color_dim: int = 2,
    ) -> Image.Image:
        """
        Renders pointcloud to an array with 3 channels appropriate for viewing as an image. The image is color coded
        according the color_dim dimension of points (typically the height).
        :param canvas_size: (width, height). Size of the canvas on which to render the image.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param color_dim: The dimension of the points to be visualized as color. Default is 2 for height.
        :return: A Image instance.
        """
        # Apply desired transformation to the point cloud. (height is here considered independent of the view).
        heights = self.points[2, :]
        points = view_points(self.points[:3, :], view, normalize=False)
        points[2, :] = heights

        # Remove points that fall outside the canvas.
        mask = np.ones(points.shape[1], dtype=bool)  # type: ignore
        mask = np.logical_and(mask, points[0, :] < canvas_size[0] - 1)
        mask = np.logical_and(mask, points[0, :] > 0)
        mask = np.logical_and(mask, points[1, :] < canvas_size[1] - 1)
        mask = np.logical_and(mask, points[1, :] > 0)
        points = points[:, mask]

        # Scale color_values to be between 0 and 255.
        color_values = points[color_dim, :]
        color_values = 255.0 * (color_values - np.amin(color_values)) / (np.amax(color_values) - np.amin(color_values))

        # Rounds to ints and generate colors that will be used in the image.
        points = np.int16(np.round(points[:2, :]))
        color_values = np.int16(np.round(color_values))
        cmap = [cm.jet(i / 255, bytes=True)[:3] for i in range(256)]

        # Populate canvas, use maximum color_value for each bin
        render = np.tile(np.expand_dims(np.zeros(canvas_size, dtype=np.uint8), axis=2), [1, 1, 3])  # type: ignore
        color_value_array: npt.NDArray[np.float64] = -1 * np.ones(canvas_size, dtype=float)  # type: ignore
        for (col, row), color_value in zip(points.T, color_values.T):
            if color_value > color_value_array[row, col]:
                color_value_array[row, col] = color_value
                render[row, col] = cmap[color_value]

        return Image.fromarray(render)

    def render_height(
        self,
        ax: axes.Axes,
        view: npt.NDArray[np.float64] = np.eye(4),
        x_lim: Tuple[float, float] = (-20, 20),
        y_lim: Tuple[float, float] = (-20, 20),
        marker_size: float = 1,
    ) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by height (z-value).
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        self._render_helper(self.points[2, :], ax, view, x_lim, y_lim, marker_size)

    def render_intensity(
        self,
        ax: axes.Axes,
        view: npt.NDArray[np.float64] = np.eye(4),
        x_lim: Tuple[float, float] = (-20, 20),
        y_lim: Tuple[float, float] = (-20, 20),
        marker_size: float = 1,
    ) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by intensity.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        self._render_helper(self.points[3, :], ax, view, x_lim, y_lim, marker_size)

    def render_label(
        self,
        ax: axes.Axes,
        id2color: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        view: npt.NDArray[np.float64] = np.eye(4),
        x_lim: Tuple[float, float] = (-20, 20),
        y_lim: Tuple[float, float] = (-20, 20),
        marker_size: float = 1.0,
    ) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points. Each points is colored based
        on labels through the label color mapping, If no mapping provided, we use the rainbow function to assign
        the colors.
        :param id2color: {label_id : (R, G, B, A)}. Id to color mapping where RGBA is within [0, 255].
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        # Label ids are expected to be concatenated at the end of the point feature.
        label = self.points[-1]
        colors: Dict[int, Tuple[Any, ...]] = {}
        if id2color is None:
            unique_label = np.unique(label)  # type: ignore
            color_rainbow = rainbow(len(unique_label), normalized=True)
            for label_id, c in zip(unique_label, color_rainbow):
                colors[label_id] = c
        else:
            for key, color in id2color.items():
                # The expected color info for matlab is normalized
                colors[key] = np.array(color) / 255.0  # type: ignore
        # Transparent if not in the id list.
        color_list = list(map(lambda x: colors.get(x, np.array((1.0, 1.0, 1.0, 0.0))), label))  # type: ignore

        self._render_helper(color_list, ax, view, x_lim, y_lim, marker_size)  # type: ignore

    def _render_helper(
        self,
        colors: Union[npt.NDArray[np.float64], List[npt.NDArray[np.float64]]],
        ax: axes.Axes,
        view: npt.NDArray[np.float64],
        x_lim: Tuple[float, float],
        y_lim: Tuple[float, float],
        marker_size: float,
    ) -> None:
        """
        Helper function for rendering.
        :param colors: Array-like or list of colors or color input for scatter function.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        points = view_points(self.points[:3, :], view, normalize=False)
        ax.scatter(points[0, :], points[1, :], c=colors, s=marker_size)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)


@dataclass
class TrajectorySampling:
    """
    Trajectory sampling config. The variables are set as optional, to make sure we can deduce last variable if only
        two are set.
    """

    # Number of poses in trajectory in addition to initial state
    num_poses: Optional[int] = None
    # [s] the time horizon of a trajectory
    time_horizon: Optional[float] = None
    # [s] length of an interval between two states
    interval_length: Optional[float] = None

    def __post_init__(self) -> None:
        """
        Make sure all entries are correctly initialized.
        """
        if self.num_poses and not isinstance(self.num_poses, int):
            raise ValueError(f"num_poses was defined but it is not int. Instead {type(self.num_poses)}!")
        if self.time_horizon:
            self.time_horizon = float(self.time_horizon)
        if self.interval_length:
            self.interval_length = float(self.interval_length)
        if self.num_poses and self.time_horizon and not self.interval_length:
            self.interval_length = self.time_horizon / self.num_poses
        elif self.num_poses and self.interval_length and not self.time_horizon:
            self.time_horizon = self.num_poses * self.interval_length
        elif self.time_horizon and self.interval_length and not self.num_poses:
            remainder = math.fmod(self.time_horizon, self.interval_length)
            is_close_to_zero = math.isclose(remainder, 0, abs_tol=PROXIMITY_ABS_TOL)
            is_close_to_interval_length = math.isclose(remainder, self.interval_length, abs_tol=PROXIMITY_ABS_TOL)
            if not is_close_to_zero and not is_close_to_interval_length:
                raise ValueError(
                    "The time horizon must be a multiple of interval length! "
                    f"time_horizon = {self.time_horizon}, interval = {self.interval_length} and is {remainder}"
                )
            self.num_poses = int(self.time_horizon / self.interval_length)
        elif self.num_poses and self.time_horizon and self.interval_length:
            if not math.isclose(self.num_poses, self.time_horizon / self.interval_length, abs_tol=PROXIMITY_ABS_TOL):
                raise ValueError(
                    "Not valid initialization of sampling class!"
                    f"time_horizon = {self.time_horizon}, "
                    f"interval = {self.interval_length}, num_poses = {self.num_poses}"
                )

        else:
            raise ValueError(
                f"Cant initialize class! num_poses = {self.num_poses}, "
                f"interval = {self.interval_length}, time_horizon = {self.time_horizon}"
            )

    @property
    def step_time(self) -> float:
        """
        :return: [s] The time difference between two poses.
        """
        if not self.interval_length:
            raise RuntimeError("Invalid interval length!")
        return self.interval_length

    def __hash__(self) -> int:
        """
        :return: hash for the dataclass. It has to be custom because the dataclass is not frozen.
            It is not frozen because we deduce the missing parameters.
        """
        return hash((self.num_poses, self.time_horizon, self.interval_length))

    def __eq__(self, other: object) -> bool:
        """
        Compare two instances of trajectory sampling
        :param other: object, needs to be TrajectorySampling class
        :return: true, if they are equal, false otherwise
        """
        if not isinstance(other, TrajectorySampling):
            return NotImplemented
        return (
            math.isclose(cast(float, other.time_horizon), cast(float, self.time_horizon))
            and math.isclose(cast(float, other.interval_length), cast(float, self.interval_length))
            and other.num_poses == self.num_poses
        )

